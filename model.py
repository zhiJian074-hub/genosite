import torch
import torch.nn as nn
from transformers import AutoModel
from utils.util_functions import cos_sim, kl_loss

# ===============================================================
# BridgeTower: 基础组件
# ===============================================================

class BridgeLayer(nn.Module):
    """
    最简稳健的桥接：Add (+) 后接 LayerNorm。
    用于把“该层的单模态语义”注入到当前跨模态表征中。
    """
    def __init__(self, d_model, use_ln=True):
        super().__init__()
        self.ln = nn.LayerNorm(d_model) if use_ln else nn.Identity()

    def forward(self, x_cross, y_uni):
        """
        x_cross: (B, Lq, d)  当前跨模态状态（Query所在流）
        y_uni:   (B, Lq或1, d) 外来单模态注入（可自动按长度广播）
        """
        if y_uni.size(1) == 1 and x_cross.size(1) > 1:
            y_uni = y_uni.expand(-1, x_cross.size(1), -1)
        return self.ln(x_cross + y_uni)


class TriModalCrossBlock(nn.Module):
    """
    三模态交互块（以序列AA tokens为Query；Text与gLM为Key/Value）
    顺序：Seq自注意力 → Seq<-Text跨注意力 → Seq<-gLM跨注意力 → FFN
    """
    def __init__(self, d_model=768, nhead=8, dropout=0.1):
        super().__init__()
        self.self_attn_seq  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_txt = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_glm = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)

    def forward(self, seq_tokens, text_tokens, glm_token):
        s = self.ln1(seq_tokens + self.self_attn_seq(seq_tokens, seq_tokens, seq_tokens, need_weights=False)[0])
        s = self.ln2(s + self.cross_attn_txt(s, text_tokens, text_tokens, need_weights=False)[0])
        s = self.ln3(s + self.cross_attn_glm(s, glm_token, glm_token, need_weights=False)[0])
        s = self.ln4(s + self.ffn(s))
        return s


# ===============================================================
# Post-Fusion MetaAdapter（非线性交互 + 向量门控）
# ===============================================================

class MetaAdapterFusionLayer(nn.Module):
    """
    Gated Bi-Modal Interaction Fusion
    在 MMSite 的 plm_dim=768 与 gLM 的 glm_dim=1280 间建立非线性交互关系。
    输出维度保持 plm_dim 以兼容下游与元学习。
    """
    def __init__(self, plm_dim: int, glm_dim: int, hidden_dim: int = 512, residual_scale: float = 0.1):
        super().__init__()
        self.plm_dim = plm_dim
        self.glm_dim = glm_dim
        self.residual_scale = residual_scale

        self.adapter_plm = nn.Sequential(
            nn.Linear(plm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, plm_dim)
        )
        self.adapter_glm = nn.Sequential(
            nn.Linear(glm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, plm_dim)
        )

        # 非线性交互项
        self.cross_layer = nn.Linear(plm_dim, plm_dim)

        # 向量级门控
        self.gate = nn.Linear(plm_dim * 2, plm_dim)

        # 稳定层
        self.norm = nn.LayerNorm(plm_dim)

        # 初始化 gate 偏置为 0，使初始 α≈0.5 更稳
        nn.init.zeros_(self.gate.bias)

    def forward(self, em_plm, em_glm):
        """
        em_plm: [B, 768]  - MMSite 融合表示
        em_glm: [B, 1280] - gLM embedding（protein-level）
        """
        em_glm = em_glm.to(em_plm.device)

        p = self.adapter_plm(em_plm)  # (B,768)
        g = self.adapter_glm(em_glm)  # (B,768)

        z = torch.sigmoid(self.gate(torch.cat([p, g], dim=-1)))   # (B,768)
        cross = torch.tanh(self.cross_layer(p * g))               # (B,768)

        fused = z * p + (1 - z) * cross
        fused = fused + self.residual_scale * em_plm
        fused = self.norm(fused)

        alpha_mean = z.mean(dim=-1)  # 日志可视化
        return fused, alpha_mean


# ===============================================================
# 模型主体：AP_align_fuse（BridgeTower + MMSite + gLM）
# ===============================================================

class AP_align_fuse(nn.Module):
    """
    1) MMSite：文本/序列的共享对齐（STEnc）+ Fusion Attention（稳定器）
    2) BridgeTower@Shared（4层）：逐层 Bridge(seq/text/gLM) + Tri-Modal Cross
    3) Post-Fusion：MetaAdapter 与 gLM 非线性后融合（元学习入口）
    4) BridgeTower@Token（2层）：每层后再次注入 gLM（影响残基级）
    """
    def __init__(self, config, hidden_size=256):
        super().__init__()
        self.config = config
        self.tau = config.tau

        # 预训练单模态编码器（冻结）
        self.text_model = AutoModel.from_pretrained(
            f'{config.model.model_dir}/{config.model.pubmed_version}'
        )
        self.seq_model = AutoModel.from_pretrained(
            f'{config.model.model_dir}/{config.model.esm_version}'
        )
        for p in self.text_model.parameters(): p.requires_grad = False
        for p in self.seq_model.parameters():  p.requires_grad = False

        # 维度设置
        self.plm_dim_seq = 1280   # ESM token hidden
        self.fusion_dim  = 768    # 对齐/融合共享维度

        self.num_attr = 17
        self.function_len = 128
        self.num_labels = 2

        # ===== 文本/序列编码与共享对齐（保留 MMSite 设计） =====
        self.project = nn.Linear(self.plm_dim_seq, self.fusion_dim)

        self.texts_encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=4, dropout=0.1)
             for _ in range(self.num_attr)]
        )
        self.text_suffix_encoder = nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=4, dropout=0.1)
        self.text_suffix_transformer = nn.TransformerEncoder(self.text_suffix_encoder, num_layers=2)
        self.text_crosses = nn.ModuleList(
            [nn.MultiheadAttention(self.fusion_dim, num_heads=4, dropout=0.1, batch_first=True)
             for _ in range(4)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.fusion_dim) for _ in range(4)])

        self.seq_suffix_encoder = nn.TransformerEncoderLayer(d_model=self.plm_dim_seq, nhead=4, dropout=0.1)
        self.seq_suffix_transformer = nn.TransformerEncoder(self.seq_suffix_encoder, num_layers=2)

        # 共享 Transformer（4层）+ Fusion Attention（稳定器）
        self.share_encoder = nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=8, dropout=0.1)
        self.share_transformer = nn.TransformerEncoder(self.share_encoder, num_layers=4)
        self.fusion_cross = nn.MultiheadAttention(self.fusion_dim, num_heads=4, dropout=0.1, batch_first=True)

        # ===== Token 头（保留你的实现） =====
        self.token_suffix_encoder_res = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim + self.plm_dim_seq, nhead=4, dropout=0.1
        )
        self.token_suffix_transformer_res = nn.TransformerEncoder(self.token_suffix_encoder_res, num_layers=2)
        self.bn2_res = nn.BatchNorm1d(hidden_size)
        self.fc2_res = nn.Linear(self.fusion_dim + self.plm_dim_seq, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(self.fusion_dim, hidden_size)
        self.classifier_token = nn.Linear(hidden_size, 2)

        # ===== 后融合 Meta-Adapter（元学习入口） =====
        self.meta_fusion_post = MetaAdapterFusionLayer(
            plm_dim=self.fusion_dim, glm_dim=1280, hidden_dim=512, residual_scale=0.1
        )

        # ===== BridgeTower 新增：多层桥接组件 =====
        # Shared（4层）
        self.k_shared_layers = 4
        self.shared_bridges_seq  = nn.ModuleList([BridgeLayer(self.fusion_dim) for _ in range(self.k_shared_layers)])
        self.shared_bridges_text = nn.ModuleList([BridgeLayer(self.fusion_dim) for _ in range(self.k_shared_layers)])
        self.shared_bridges_glm  = nn.ModuleList([BridgeLayer(self.fusion_dim) for _ in range(self.k_shared_layers)])
        self.shared_cross_blocks = nn.ModuleList(
            [TriModalCrossBlock(d_model=self.fusion_dim, nhead=8, dropout=0.1) for _ in range(self.k_shared_layers)]
        )
        self.proj_glm_to_fusion  = nn.Linear(1280, self.fusion_dim)  # gLM 1280 → 768

        # Token 后缀（2层）
        self.k_token_layers = 2
        self.token_bridges_glm = nn.ModuleList(
            [BridgeLayer(self.fusion_dim + self.plm_dim_seq) for _ in range(self.k_token_layers)]
        )

    # ===========================================================
    # 一些公共子函数（与原始保持一致接口）
    # ===========================================================

    def _get_similarity(self, image_features, text_features):
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-12)
        text_features  = text_features  / (text_features.norm(dim=1, keepdim=True)  + 1e-12)
        logits_per_image = image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()

    def _softlabel_loss_3d(self, seq_features, text_features, tau):
        """
        用 shared 段输出进行软标签对齐（与 MMSite 一致）。
        """
        seq_features  = seq_features.mean(dim=1)
        text_features = text_features.mean(dim=1)
        seq_sim, text_sim = cos_sim(seq_features, seq_features), cos_sim(text_features, text_features)
        lp_img, lp_txt = self._get_similarity(seq_features, text_features)
        return (kl_loss(lp_img, seq_sim, tau=tau) + kl_loss(lp_txt, text_sim, tau=tau)) / 2.0

    def _get_model_output(self, input_ids, attention_mask, modal):
        model = self.text_model if modal == 'text' else self.seq_model
        return model(input_ids=input_ids, attention_mask=attention_mask)

    def _get_text_branch_output(self, text_outputs):
        """
        MACross + suffix transformer（与你原实现一致）
        """
        texts_output = [self.texts_encoder[i](text_outputs[i][0]) for i in range(self.num_attr)]
        texts_output_cls = [texts_output[idx][:, 0, :].unsqueeze(1) for idx in range(len(texts_output)) if idx != 3]
        texts_output_cls = torch.cat(texts_output_cls, dim=1)
        texts_output_cls = self.text_suffix_transformer(texts_output_cls)
        text_func = texts_output[3]
        x = texts_output_cls
        for i in range(4):
            _x = x
            x  = self.text_crosses[i](x, text_func, text_func)
            x  = self.norms[i](x[0] + _x)
        return x  # (B, M, 768)

    # ===========================================================
    # 前向：BridgeTower@Shared + Fusion + post-fusion + BridgeTower@Token
    # ===========================================================

    def forward(self,
                anchor_text_input_ids=None, anchor_text_attention_mask=None,
                anchor_seq_input_ids=None,  anchor_seq_attention_mask=None,
                entry_ids=None, test=False):

        device = anchor_seq_input_ids.device

        # ---- 文本侧 ----
        if not test:
            text_list = []
            for i in range(len(anchor_text_input_ids)):
                text_list.append(self._get_model_output(anchor_text_input_ids[i],
                                                        anchor_text_attention_mask[i], 'text'))
        else:
            text_out = self._get_model_output(anchor_text_input_ids, anchor_text_attention_mask, 'text')

        # ---- 序列侧（AA token）----
        seq_out = self._get_model_output(anchor_seq_input_ids, anchor_seq_attention_mask, 'seq')
        aa_hidden = seq_out[0]  # (B, L, 1280)

        # ---- 文本支路聚合（与你一致）----
        if not test:
            text_branch = self._get_text_branch_output(text_list)  # (B,M,768)
        else:
            text_branch = text_out[0]

        # ---- gLM 向量（查表/占位）----
        if entry_ids is not None and hasattr(self, "glm_lookup"):
            glm_vec = self.glm_lookup(entry_ids)
            if not isinstance(glm_vec, torch.Tensor):
                glm_vec = torch.tensor(glm_vec, dtype=aa_hidden.dtype)
            glm_vec = glm_vec.to(device)
        else:
            glm_vec = torch.zeros((aa_hidden.size(0), 1280), dtype=aa_hidden.dtype, device=device)

        # =======================================================
        # BridgeTower@Shared：逐层桥接 + 三模态交互（4层）
        # =======================================================
        seq_mean = aa_hidden.mean(dim=1)                 # (B,1280)
        x_seq  = self.project(seq_mean).unsqueeze(1)     # (B,1,768)
        x_text = text_branch                              # (B,M,768)
        x_glm  = self.proj_glm_to_fusion(glm_vec).unsqueeze(1)  # (B,1,768)

        for t in range(self.k_shared_layers):
            # Add&Norm 桥接：把该层单模态语义注入当前跨模态状态
            # 这里对 text 用 mean-pool 注入，可按需替换成更细粒度策略
            x_seq  = self.shared_bridges_seq[t](x_seq,  x_seq)   # 自桥，形式统一（也可去掉）
            x_seq  = self.shared_bridges_text[t](x_seq, x_text.mean(dim=1, keepdim=True))
            x_seq  = self.shared_bridges_glm[t](x_seq,  x_glm)
            # 三模态交互：Seq 为 Q，Text & gLM 为 KV
            x_seq = self.shared_cross_blocks[t](x_seq, x_text, x_glm)

        # =======================================================
        # Fusion Attention（保留你的稳定器作为“第0层门”）
        # =======================================================
        seq_shared  = x_seq
        text_shared = x_text
        fusion_tensor, _ = self.fusion_cross(seq_shared, text_shared, text_shared)  # (B,1,768)
        fusion_vec = fusion_tensor.squeeze(1)                                       # (B,768)

        # =======================================================
        # Post-Fusion：MetaAdapter 与 gLM 非线性后融合（元学习入口）
        # =======================================================
        fusion_vec_fused, alpha = self.meta_fusion_post(fusion_vec, glm_vec)        # (B,768), (B,)

        # =======================================================
        # BridgeTower@Token：在 token 后缀每层后再桥接注入 gLM
        # =======================================================
        aa_ctx = self.seq_suffix_transformer(aa_hidden)  # (B, L, 1280)
        fusion_expanded = fusion_vec_fused.unsqueeze(1).expand(-1, aa_ctx.size(1), -1)  # (B,L,768)
        x_tok = torch.cat([aa_ctx, fusion_expanded], dim=-1)  # (B, L, 1280+768)

        glm_tok = self.proj_glm_to_fusion(glm_vec).unsqueeze(1).expand(-1, x_tok.size(1), -1)  # (B,L,768)
        for t in range(self.k_token_layers):
            x_tok = self.token_suffix_transformer_res.layers[t](x_tok)
            # y_uni 必须和 x_tok 维度一致：2048 = 1280(AA) + 768(fusion)
            # 在 AA 部分填 0，在 fusion 部分填入 glm_tok
            g_pad = torch.zeros_like(x_tok)
            g_pad[..., self.plm_dim_seq:] = glm_tok  # 把后 768 段替换为 glm_tok
            x_tok = self.token_bridges_glm[t](x_tok, g_pad)

        # =======================================================
        # Token 级预测头（与你一致）
        # =======================================================
        h = x_tok
        h = torch.relu(self.bn2_res(self.fc2_res(h).permute(0, 2, 1)).permute(0, 2, 1))
        token_logits = self.classifier_token(h)[:, :, self.num_labels - 1].squeeze(-1)

        # 对齐阶段的软标签对齐损失（seq_shared/text_shared）
        cl_loss = self._softlabel_loss_3d(seq_shared, text_shared, tau=self.tau)

        return {
            "token_logits": torch.sigmoid(token_logits),
            "contrastive_loss": cl_loss,
            "alpha": alpha.mean().item() if isinstance(alpha, torch.Tensor) else float(alpha),
        }
