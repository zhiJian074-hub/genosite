import torch
import torch.nn as nn
from transformers import AutoModel
from utils.util_functions import cos_sim, kl_loss

# ===============================================================
# BridgeTower: core components
# ===============================================================

class BridgeLayer(nn.Module):
    """
    A simple, stable bridge: addition followed by LayerNorm.
    Injects layer-specific unimodal semantics into the current cross-modal representation.
    """
    def __init__(self, d_model, use_ln=True):
        super().__init__()
        self.ln = nn.LayerNorm(d_model) if use_ln else nn.Identity()

    def forward(self, x_cross, y_uni):
        """
        x_cross: (B, Lq, d)  Current cross-modal state (the query stream)
        y_uni:   (B, Lq or 1, d) Incoming unimodal features (broadcast across length when needed)
        """
        if y_uni.size(1) == 1 and x_cross.size(1) > 1:
            y_uni = y_uni.expand(-1, x_cross.size(1), -1)
        return self.ln(x_cross + y_uni)


class TriModalCrossBlock(nn.Module):
    """
    Trimodal interaction block (AA sequence tokens as queries; text and gLM as keys/values)
    Order: sequence self-attention -> sequence-to-text cross-attention -> sequence-to-gLM cross-attention -> FFN
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
# Post-fusion MetaAdapter (nonlinear interaction and vector gating)
# ===============================================================

class MetaAdapterFusionLayer(nn.Module):
    """
    Gated Bi-Modal Interaction Fusion
    Models nonlinear interactions between the MMSite representation (plm_dim=768) and the gLM representation (glm_dim=1280).
    Keeps the output dimension at plm_dim for downstream processing and meta-learning.
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

        # Nonlinear interaction term
        self.cross_layer = nn.Linear(plm_dim, plm_dim)

        # Vector-level gate
        self.gate = nn.Linear(plm_dim * 2, plm_dim)

        # Normalization for stability
        self.norm = nn.LayerNorm(plm_dim)

        # Initialize the gate bias to zero so the initial alpha is approximately 0.5
        nn.init.zeros_(self.gate.bias)

    def forward(self, em_plm, em_glm):
        """
        em_plm: [B, 768]  - MMSite fused representation
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

        alpha_mean = z.mean(dim=-1)  # Logged for visualization
        return fused, alpha_mean


# ===============================================================
# Main model: AP_align_fuse (BridgeTower + MMSite + gLM)
# ===============================================================

class AP_align_fuse(nn.Module):
    """
    1) MMSite: shared text/sequence alignment (STEnc) and Fusion Attention (stabilizer)
    2) BridgeTower@Shared (four layers): per-layer sequence/text/gLM bridges and trimodal cross-attention
    3) Post-fusion: nonlinear gLM fusion through MetaAdapter (meta-learning entry point)
    4) BridgeTower@Token (two layers): inject gLM after each layer to affect residue-level features
    """
    def __init__(self, config, hidden_size=256):
        super().__init__()
        self.config = config
        self.tau = config.tau

        # Frozen pretrained unimodal encoders
        self.text_model = AutoModel.from_pretrained(
            f'{config.model.model_dir}/{config.model.pubmed_version}'
        )
        self.seq_model = AutoModel.from_pretrained(
            f'{config.model.model_dir}/{config.model.esm_version}'
        )
        for p in self.text_model.parameters(): p.requires_grad = False
        for p in self.seq_model.parameters():  p.requires_grad = False

        # Dimension settings
        self.plm_dim_seq = 1280   # ESM token hidden
        self.fusion_dim  = 768    # Shared alignment and fusion dimension

        self.num_attr = 17
        self.function_len = 128
        self.num_labels = 2

        # ===== Text/sequence encoding and shared alignment (retaining the MMSite design) =====
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

        # Four-layer shared Transformer and Fusion Attention stabilizer
        self.share_encoder = nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=8, dropout=0.1)
        self.share_transformer = nn.TransformerEncoder(self.share_encoder, num_layers=4)
        self.fusion_cross = nn.MultiheadAttention(self.fusion_dim, num_heads=4, dropout=0.1, batch_first=True)

        # ===== Token prediction head (retaining the existing implementation) =====
        self.token_suffix_encoder_res = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim + self.plm_dim_seq, nhead=4, dropout=0.1
        )
        self.token_suffix_transformer_res = nn.TransformerEncoder(self.token_suffix_encoder_res, num_layers=2)
        self.bn2_res = nn.BatchNorm1d(hidden_size)
        self.fc2_res = nn.Linear(self.fusion_dim + self.plm_dim_seq, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(self.fusion_dim, hidden_size)
        self.classifier_token = nn.Linear(hidden_size, 2)

        # ===== Post-fusion MetaAdapter (meta-learning entry point) =====
        self.meta_fusion_post = MetaAdapterFusionLayer(
            plm_dim=self.fusion_dim, glm_dim=1280, hidden_dim=512, residual_scale=0.1
        )

        # ===== Additional BridgeTower components: multi-layer bridges =====
        # Shared stage (four layers)
        self.k_shared_layers = 4
        self.shared_bridges_seq  = nn.ModuleList([BridgeLayer(self.fusion_dim) for _ in range(self.k_shared_layers)])
        self.shared_bridges_text = nn.ModuleList([BridgeLayer(self.fusion_dim) for _ in range(self.k_shared_layers)])
        self.shared_bridges_glm  = nn.ModuleList([BridgeLayer(self.fusion_dim) for _ in range(self.k_shared_layers)])
        self.shared_cross_blocks = nn.ModuleList(
            [TriModalCrossBlock(d_model=self.fusion_dim, nhead=8, dropout=0.1) for _ in range(self.k_shared_layers)]
        )
        self.proj_glm_to_fusion  = nn.Linear(1280, self.fusion_dim)  # gLM 1280 → 768

        # Token suffix (two layers)
        self.k_token_layers = 2
        self.token_bridges_glm = nn.ModuleList(
            [BridgeLayer(self.fusion_dim + self.plm_dim_seq) for _ in range(self.k_token_layers)]
        )

    # ===========================================================
    # Shared helper methods (preserving the original interfaces)
    # ===========================================================

    def _get_similarity(self, image_features, text_features):
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-12)
        text_features  = text_features  / (text_features.norm(dim=1, keepdim=True)  + 1e-12)
        logits_per_image = image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()

    def _softlabel_loss_3d(self, seq_features, text_features, tau):
        """
        Align soft labels using the output of the shared stage, as in MMSite.
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
        MACross and suffix Transformer (consistent with the existing implementation)
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
    # Forward pass: BridgeTower@Shared, fusion, post-fusion, and BridgeTower@Token
    # ===========================================================

    def forward(self,
                anchor_text_input_ids=None, anchor_text_attention_mask=None,
                anchor_seq_input_ids=None,  anchor_seq_attention_mask=None,
                entry_ids=None, test=False):

        device = anchor_seq_input_ids.device

        # ---- Text branch ----
        if not test:
            text_list = []
            for i in range(len(anchor_text_input_ids)):
                text_list.append(self._get_model_output(anchor_text_input_ids[i],
                                                        anchor_text_attention_mask[i], 'text'))
        else:
            text_out = self._get_model_output(anchor_text_input_ids, anchor_text_attention_mask, 'text')

        # ---- Sequence branch (AA tokens)----
        seq_out = self._get_model_output(anchor_seq_input_ids, anchor_seq_attention_mask, 'seq')
        aa_hidden = seq_out[0]  # (B, L, 1280)

        # ---- Text-branch aggregation (consistent with the existing implementation)----
        if not test:
            text_branch = self._get_text_branch_output(text_list)  # (B,M,768)
        else:
            text_branch = text_out[0]

        # ---- gLM vector (lookup or placeholder)----
        if entry_ids is not None and hasattr(self, "glm_lookup"):
            glm_vec = self.glm_lookup(entry_ids)
            if not isinstance(glm_vec, torch.Tensor):
                glm_vec = torch.tensor(glm_vec, dtype=aa_hidden.dtype)
            glm_vec = glm_vec.to(device)
        else:
            glm_vec = torch.zeros((aa_hidden.size(0), 1280), dtype=aa_hidden.dtype, device=device)

        # =======================================================
        # BridgeTower@Shared: per-layer bridges and trimodal interaction (four layers)
        # =======================================================
        seq_mean = aa_hidden.mean(dim=1)                 # (B,1280)
        x_seq  = self.project(seq_mean).unsqueeze(1)     # (B,1,768)
        x_text = text_branch                              # (B,M,768)
        x_glm  = self.proj_glm_to_fusion(glm_vec).unsqueeze(1)  # (B,1,768)

        for t in range(self.k_shared_layers):
            # Add-and-normalize bridge: inject unimodal semantics into the current cross-modal state
            # Inject mean-pooled text here; this can be replaced with a finer-grained strategy
            x_seq  = self.shared_bridges_seq[t](x_seq,  x_seq)   # Self-bridge for a uniform structure (optional)
            x_seq  = self.shared_bridges_text[t](x_seq, x_text.mean(dim=1, keepdim=True))
            x_seq  = self.shared_bridges_glm[t](x_seq,  x_glm)
            # Trimodal interaction: sequence as queries, text and gLM as keys/values
            x_seq = self.shared_cross_blocks[t](x_seq, x_text, x_glm)

        # =======================================================
        # Fusion Attention (retaining the stabilizer as a preliminary gate)
        # =======================================================
        seq_shared  = x_seq
        text_shared = x_text
        fusion_tensor, _ = self.fusion_cross(seq_shared, text_shared, text_shared)  # (B,1,768)
        fusion_vec = fusion_tensor.squeeze(1)                                       # (B,768)

        # =======================================================
        # Post-fusion: nonlinear gLM fusion through MetaAdapter (meta-learning entry point)
        # =======================================================
        fusion_vec_fused, alpha = self.meta_fusion_post(fusion_vec, glm_vec)        # (B,768), (B,)

        # =======================================================
        # BridgeTower@Token: inject gLM through a bridge after each token-suffix layer
        # =======================================================
        aa_ctx = self.seq_suffix_transformer(aa_hidden)  # (B, L, 1280)
        fusion_expanded = fusion_vec_fused.unsqueeze(1).expand(-1, aa_ctx.size(1), -1)  # (B,L,768)
        x_tok = torch.cat([aa_ctx, fusion_expanded], dim=-1)  # (B, L, 1280+768)

        glm_tok = self.proj_glm_to_fusion(glm_vec).unsqueeze(1).expand(-1, x_tok.size(1), -1)  # (B,L,768)
        for t in range(self.k_token_layers):
            x_tok = self.token_suffix_transformer_res.layers[t](x_tok)
            # y_uni must match x_tok: 2048 = 1280 (AA) + 768 (fusion)
            # Fill the AA portion with zeros and the fusion portion with glm_tok
            g_pad = torch.zeros_like(x_tok)
            g_pad[..., self.plm_dim_seq:] = glm_tok  # Place glm_tok in the final 768 dimensions
            x_tok = self.token_bridges_glm[t](x_tok, g_pad)

        # =======================================================
        # Token-level prediction head (consistent with the existing implementation)
        # =======================================================
        h = x_tok
        h = torch.relu(self.bn2_res(self.fc2_res(h).permute(0, 2, 1)).permute(0, 2, 1))
        token_logits = self.classifier_token(h)[:, :, self.num_labels - 1].squeeze(-1)

        # Soft-label alignment loss for the alignment stage (seq_shared/text_shared)
        cl_loss = self._softlabel_loss_3d(seq_shared, text_shared, tau=self.tau)

        return {
            "token_logits": torch.sigmoid(token_logits),
            "contrastive_loss": cl_loss,
            "alpha": alpha.mean().item() if isinstance(alpha, torch.Tensor) else float(alpha),
        }
