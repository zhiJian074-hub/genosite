import os, sys, argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

proj_path = os.path.abspath('.')
sys.path.append(proj_path)

from utils.util_functions import read_config, log_metrics, measure, prepare, get_model
from utils.util_classes import MyDataset, WarmupCosineAnnealingLR, GLMEmbeddingLookup

device, pprint = None, None
import torch.nn.functional as F

# ============================================================
# Fusion parameter name patterns (including BridgeTower and MetaAdapter)
# ============================================================
_FUSION_KEYS = [
    # === Retain possible legacy names ===
    "fusion_ctx", "proj_glm", "meta_adapter", "fusion_head",
    "post_fuse", "glm_gate", "glm_mlp",

    # === Actual parameter names used here ===
    "meta_fusion_post",     # Post-fusion MetaAdapter (meta-learning entry point)

    # === BridgeTower components ===
    "shared_bridges_seq", "shared_bridges_text", "shared_bridges_glm",
    "shared_cross_blocks", "token_bridges_glm", "proj_glm_to_fusion",

    # === Nonlinear interaction layer and vector gate within MetaAdapter
    "cross_layer", "gate",
    # === Fallback patterns for renamed parameters
    "fusion_adapter", "fusion_layer", "fusion_block", "fusion_gate_layer", "fusion_linear", "glm_adapter",
]

def _is_fusion_param(name: str) -> bool:
    return any(k in name for k in _FUSION_KEYS)

def freeze_all_except_fusion(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad = _is_fusion_param(n)

def get_fusion_param_state(model: nn.Module):
    """Extract only the fusion-related subset of the state_dict."""
    sd = model.state_dict()
    return {k: v.detach().clone() for k, v in sd.items() if _is_fusion_param(k)}

@torch.no_grad()
def load_fusion_param_state_(model: nn.Module, sub_state: dict):
    """Restore the fusion-related parameter subset."""
    for k, v in sub_state.items():
        if k in model.state_dict():
            model.state_dict()[k].copy_(v)

@torch.no_grad()
def reptile_update_(model: nn.Module, old_sub: dict, step_sub: dict, beta: float):
    """Reptile: θ ← (1-β)θ + βθ′"""
    for k, v_new in step_sub.items():
        if k in old_sub:
            p = model.state_dict()[k]
            p.copy_((1.0 - beta) * old_sub[k] + beta * v_new)


# ============================================================
# Lightweight MAML/Reptile-style meta-learning (post-fusion)
# ============================================================
def meta_train_post_fusion_maml(model: nn.Module,
                                train_loader,
                                config,
                                device,
                                pprint,
                                inner_steps: int = 1,
                                beta: float = 0.5,
                                bce_weight: float = 1.0,
                                align_weight: float = 0.0,
                                ortho_weight: float = 0.0):
    """
    Post-fusion meta-learning stage (using the actual metric names)
    - Split each batch into support and query sets
    - Inner-loop: Update fusion parameters on the support set
    - Outer-loop: Reptile-style update
    - Record training F1 each epoch and save the best model
    """

    pprint("Stage 2.5: Meta-learning (post-fusion MAML/Reptile) ...")
    model.to(device)
    model.train()
    freeze_all_except_fusion(model)

    inner_opt = optim.SGD(
        [p for n, p in model.named_parameters() if p.requires_grad and _is_fusion_param(n)],
        lr=float(config.meta.inner_lr),
        momentum=0.0
    )

    bce = nn.BCELoss()
    epochs = int(config.meta.meta_epochs)

    best_f1, best_metrics, ckpt_meta = 0.0, {}, None

    for ep in range(epochs):
        tr = {m: 0.0 for m in config.train.metrics}
        for step, batch in enumerate(train_loader):
            text_list, seq, labels, length, meta_b = batch
            labels = labels.to(device)
            entry_ids = meta_b["Entry"]
            B = labels.size(0)
            split = max(1, B // 2)

            def _slice_texts(texts, sl):
                return [{"input_ids": be["input_ids"][sl].to(device),
                         "attention_mask": be["attention_mask"][sl].to(device)} for be in texts]

            sup_text = _slice_texts(text_list, slice(0, split))
            qry_text = _slice_texts(text_list, slice(split, B))
            sup_seq = {k: v[:split].to(device) for k, v in seq.items()}
            qry_seq = {k: v[split:].to(device) for k, v in seq.items()}
            sup_lab, qry_lab = labels[:split], labels[split:]
            sup_entry = [entry_ids[i] for i in range(split)]
            qry_entry = [entry_ids[i] for i in range(split, B)]

            old_sub = get_fusion_param_state(model)

            # Inner-loop: Learn on the support set
            for _ in range(inner_steps):
                inner_opt.zero_grad()
                out_sup = model(
                    anchor_text_input_ids=[t["input_ids"] for t in sup_text],
                    anchor_text_attention_mask=[t["attention_mask"] for t in sup_text],
                    anchor_seq_input_ids=sup_seq["input_ids"],
                    anchor_seq_attention_mask=sup_seq["attention_mask"],
                    entry_ids=sup_entry
                )
                loss_sup = bce(out_sup["token_logits"].float(), sup_lab.float()) * bce_weight
                loss_sup.backward()
                inner_opt.step()

            # Outer-loop update (Reptile)
            step_sub = get_fusion_param_state(model)
            load_fusion_param_state_(model, old_sub)
            reptile_update_(model, old_sub, step_sub, beta=float(config.meta.alpha))

            if step % 50 == 0:
                with torch.no_grad():
                    out_qry = model(
                        anchor_text_input_ids=[t["input_ids"] for t in qry_text],
                        anchor_text_attention_mask=[t["attention_mask"] for t in qry_text],
                        anchor_seq_input_ids=qry_seq["input_ids"],
                        anchor_seq_attention_mask=qry_seq["attention_mask"],
                        entry_ids=qry_entry
                    )
                    q_loss = bce(out_qry["token_logits"].float(), qry_lab.float()).item()
                    pprint(f"[Meta] ep={ep+1} step={step} | q_loss={q_loss:.6f}")

        # === Compute epoch-average metrics over the full train_loader ===
        model.eval()
        with torch.no_grad():
            for texts, seq, y, L, meta in tqdm(train_loader, desc=f"MetaEval {ep+1}/{epochs}"):
                out = model(
                    anchor_text_input_ids=[b['input_ids'].to(device) for b in texts],
                    anchor_text_attention_mask=[b['attention_mask'].to(device) for b in texts],
                    anchor_seq_input_ids=seq['input_ids'].to(device),
                    anchor_seq_attention_mask=seq['attention_mask'].to(device),
                    entry_ids=meta['Entry']
                )
                y = y.to(device)
                pred = (out['token_logits'] > 0.5).float()
                m = measure(
                    y.cpu().numpy(),
                    pred.cpu().numpy(),
                    out['token_logits'].detach().float().cpu().numpy(),
                    L
                )
                for k, v in m.items():
                    if k in tr:
                        tr[k] += v
        for k in tr:
            tr[k] /= len(train_loader)
        log_metrics(pprint, 'meta-train', ep+1, tr, 'Token')

        # === Save the best model ===
        if tr.get('f1', 0.0) > best_f1:
            best_f1, best_metrics = tr['f1'], tr
            os.makedirs(config.train.save_path, exist_ok=True)
            # Remove the previous meta-model checkpoint
            for f in os.listdir(config.train.save_path):
                if f.startswith("best_model_fuse_meta_"):
                    try:
                        os.remove(os.path.join(config.train.save_path, f))
                    except Exception as e:
                        pprint(f"[Warning] 删除旧meta模型失败: {e}")

            ckpt_meta = os.path.join(
                config.train.save_path,
                f"best_model_fuse_meta_{best_f1:.4f}.pt"
            )
            torch.save(model.state_dict(), ckpt_meta, _use_new_zipfile_serialization=False)
            pprint(f"💾 Saving best meta-fused model -> {ckpt_meta}")

        model.train()

    info = "Best meta-train (Token): " + ", ".join([f"{k}: {v:.4f}" for k, v in best_metrics.items()])
    pprint(info)
    return ckpt_meta



# ============================================================
# Data loading, alignment, fusion, evaluation, and testing
# ============================================================

def prepare_dataloaders(config):
    train_dataset = MyDataset(config, 'train')
    valid_dataset = MyDataset(config, 'valid')
    align_val_dataset = MyDataset(config, 'valid', align=True)
    test_dataset = MyDataset(config, 'test')

    # Debug subset
    if 'sample_ratio' in config['dataset']:
        ratio = config['dataset']['sample_ratio']
        if ratio < 1.0:
            from torch.utils.data import Subset
            subset_size = int(len(train_dataset) * ratio)
            train_dataset = Subset(train_dataset, range(subset_size))
            print(f"[Debug] Using {subset_size}/{len(train_dataset)} samples for training (ratio={ratio}).")

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, drop_last=True)
    val_loader   = DataLoader(valid_dataset,  batch_size=config.val.batch_size,  drop_last=True)
    test_loader  = DataLoader(test_dataset,   batch_size=config.test.batch_size, drop_last=False)
    align_valid  = DataLoader(align_val_dataset, batch_size=config.val.batch_size, drop_last=True)
    return train_loader, val_loader, align_valid, test_loader


def align(model, train_loader, align_valid_loader, config):
    epochs = config.train.align_epoch
    best_loss, best_metrics = 1e9, {}
    model.to(device)

    bce_loss = nn.BCELoss()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr)
    sch = WarmupCosineAnnealingLR(opt, total_steps=epochs * len(train_loader))

    for ep in range(epochs):
        model.train()
        tr = {m:0.0 for m in config.train.metrics}

        for texts, seq, labels, lengths, meta in tqdm(train_loader, desc=f"Align {ep+1}/{epochs}"):
            opt.zero_grad()
            out = model(
                anchor_text_input_ids=[b['input_ids'].to(device) for b in texts],
                anchor_text_attention_mask=[b['attention_mask'].to(device) for b in texts],
                anchor_seq_input_ids=seq['input_ids'].to(device),
                anchor_seq_attention_mask=seq['attention_mask'].to(device),
                entry_ids=meta['Entry'],
            )
            loss = out['contrastive_loss'].float()
            loss.backward()
            opt.step()
            sch.step()
            tr['loss'] += loss.item()

        for k in tr: tr[k] /= len(train_loader)
        log_metrics(pprint, 'train', ep+1, tr, 'Contrastive Loss')

        _, val_metrics = evaluate(epochs, ep, model, align_valid_loader, bce_loss, config, align=True)
        if val_metrics['loss'] < best_loss:
            best_loss, best_metrics = val_metrics['loss'], val_metrics

            os.makedirs(config.train.save_path, exist_ok=True)
            # Remove the previous best alignment checkpoint
            for f in os.listdir(config.train.save_path):
                if f.startswith("best_model_align_"):
                    try: os.remove(os.path.join(config.train.save_path, f))
                    except Exception as e: pprint(f"[Warning] 删除旧模型失败: {e}")

            ckpt = os.path.join(config.train.save_path, f"best_model_align_{best_loss:.6f}.pth")
            torch.save(model, ckpt)
            pprint(f"Saving best align model -> {ckpt}")

    info = "Best val (Contrastive): " + ", ".join([f"{k}: {v:.4f}" for k,v in best_metrics.items()])
    pprint(info)
    return ckpt


def meta_train(model, train_loader, meta_cfg):
    """
    Train only post-fusion meta_fusion_post (legacy interface, used when needed)
    """
    print("Stage 2.5: Meta-learning (post-fusion Meta-Adapter) ...")

    model.to(device)
    model.train()

    # Freeze all parameters except meta_fusion_post
    for p in model.parameters():
        p.requires_grad = False
    for p in model.meta_fusion_post.parameters():
        p.requires_grad = True

    bce = nn.BCELoss()
    optimizer = optim.Adam(model.meta_fusion_post.parameters(), lr=meta_cfg.inner_lr)

    epochs = meta_cfg.meta_epochs
    for ep in range(epochs):
        running = 0.0
        for step, (texts, seq, token_labels, length, meta) in enumerate(
            tqdm(train_loader, desc=f"Meta {ep+1}/{epochs}")
        ):
            optimizer.zero_grad()

            out = model(
                anchor_text_input_ids=[b['input_ids'].to(device) for b in texts],
                anchor_text_attention_mask=[b['attention_mask'].to(device) for b in texts],
                anchor_seq_input_ids=seq['input_ids'].to(device),
                anchor_seq_attention_mask=seq['attention_mask'].to(device),
                entry_ids=meta['Entry'],
            )

            token_labels = token_labels.to(device)
            meta_loss = bce(out['token_logits'].float(), token_labels.float())
            meta_loss.backward()
            optimizer.step()

            running += meta_loss.item()
            if step % 50 == 0:
                print(f"[Meta] ep={ep+1} step={step} loss={meta_loss.item():.6f} alpha≈{out['alpha']:.4f}")

        print(f"✅ Meta epoch {ep+1} avg loss: {running/len(train_loader):.6f}")

    # Restore parameter trainability
    for p in model.parameters():
        p.requires_grad = True
    print("✅ 元学习阶段完成。")


def fuse(checkpoint_align, train_loader, val_loader, config):
    model = torch.load(checkpoint_align, map_location='cpu')
    model.to(device)
    model.train()

    # Freeze meta_fusion_post to preserve the learned gate
    for p in model.meta_fusion_post.parameters():
        p.requires_grad = False

    # Train only the token head and suffix modules, as in the original paper
    suffixes = [
        'seq_suffix_encoder', 'seq_suffix_transformer',
        'token_suffix_encoder_res', 'token_suffix_transformer_res',
        'token_suffix_encoder', 'token_suffix_transformer',
        'fusion_cross', 'bn2_res', 'fc2_res', 'bn2', 'fc2', 'classifier_token'
    ]
    for n, p in model.named_parameters():
        p.requires_grad = any(sfx in n for sfx in suffixes)
    pprint("trainable params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    bce = nn.BCELoss()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr)
    sch = WarmupCosineAnnealingLR(opt, total_steps=config.train.epochs * len(train_loader))

    best_f1, best_metrics, ckpt = 0.0, {}, None
    for ep in range(config.train.epochs):
        model.train()
        tr = {m: 0.0 for m in config.train.metrics}

        for texts, seq, y, L, meta in tqdm(train_loader, desc=f"Fuse {ep+1}/{config.train.epochs}"):
            opt.zero_grad()
            out = model(
                anchor_text_input_ids=[b['input_ids'].to(device) for b in texts],
                anchor_text_attention_mask=[b['attention_mask'].to(device) for b in texts],
                anchor_seq_input_ids=seq['input_ids'].to(device),
                anchor_seq_attention_mask=seq['attention_mask'].to(device),
                entry_ids=meta['Entry'],
            )
            y = y.to(device)
            loss = bce(out['token_logits'].float(), y.float())
            loss.backward()
            opt.step()
            sch.step()
            tr['loss'] += loss.item()

            pred = (out['token_logits'] > 0.5).float()
            m = measure(
                y.cpu().numpy(),
                pred.cpu().numpy(),
                out['token_logits'].detach().float().cpu().numpy(),
                L
            )
            for k, v in m.items():
                tr[k] += v

        for k in tr:
            tr[k] /= len(train_loader)
        log_metrics(pprint, 'train', ep+1, tr, 'Token')

        val_metrics, _ = evaluate(config.train.epochs, ep, model, val_loader, bce, config, align=False)
        if val_metrics['f1'] > best_f1:
            best_f1, best_metrics = val_metrics['f1'], val_metrics

            os.makedirs(config.train.save_path, exist_ok=True)
            # Remove the previous best fusion checkpoint
            for f in os.listdir(config.train.save_path):
                if f.startswith("best_model_fuse_"):
                    try: os.remove(os.path.join(config.train.save_path, f))
                    except Exception as e: pprint(f"[Warning] 删除旧 fuse 模型失败: {e}")

            ckpt = os.path.join(config.train.save_path, f"best_model_fuse_{best_f1:.4f}.pt")
            torch.save(model.state_dict(), ckpt, _use_new_zipfile_serialization=False)
            pprint(f"Saving best fuse model -> {ckpt}")

    info = "Best val (Token): " + ", ".join([f"{k}: {v:.4f}" for k, v in best_metrics.items()])
    pprint(info)
    return ckpt


@torch.no_grad()
def evaluate(epochs, ep, model, loader, bce_loss, config, align=False):
    model.eval()
    val_tok, val_cl = {m:0.0 for m in config.train.metrics}, {m:0.0 for m in config.train.metrics}
    for texts, seq, y, L, meta in tqdm(loader, desc=f"Val {ep+1}/{epochs}"):
        if align:
            text_ids  = [b['input_ids'].to(device) for b in texts]
            text_mask = [b['attention_mask'].to(device) for b in texts]
        else:
            text_ids  = texts['input_ids'].to(device)
            text_mask = texts['attention_mask'].to(device)

        out = model(
            anchor_text_input_ids=text_ids,
            anchor_text_attention_mask=text_mask,
            anchor_seq_input_ids=seq['input_ids'].to(device),
            anchor_seq_attention_mask=seq['attention_mask'].to(device),
            entry_ids=meta['Entry'],
            test=not align
        )
        y = y.to(device)
        tok_loss = bce_loss(out['token_logits'].float(), y.float())
        val_tok['loss'] += tok_loss.item()
        val_cl['loss']  += out['contrastive_loss'].float().item()

        pred = (out['token_logits'] > 0.5).float()
        m = measure(y.cpu().numpy(), pred.cpu().numpy(),
                    out['token_logits'].detach().float().cpu().numpy(), L)
        for k,v in m.items(): val_tok[k] += v

    for k in val_tok: val_tok[k] /= len(loader)
    for k in val_cl:  val_cl[k]  /= len(loader)
    log_metrics(pprint, 'val', ep+1, val_tok, 'Token')
    log_metrics(pprint, 'val', ep+1, val_cl,  'Contrastive Loss')
    return val_tok, val_cl


@torch.no_grad()
def test(model, ckpt, loader, config):
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.to(device)
    model.eval()
    tok = {m:0.0 for m in config.train.metrics}

    # 🧩 Count the total number of samples
    total_samples = len(loader.dataset)
    pprint(f"📊 一共测试样本数: {total_samples}")

    for texts, seq, y, L, meta in tqdm(loader, desc="Testing"):
        out = model(
            anchor_text_input_ids=texts['input_ids'].to(device),
            anchor_text_attention_mask=texts['attention_mask'].to(device),
            anchor_seq_input_ids=seq['input_ids'].to(device),
            anchor_seq_attention_mask=seq['attention_mask'].to(device),
            entry_ids=meta['Entry'], test=True
        )
        y = y.to(device)
        pred = (out['token_logits'] > 0.5).float()
        m = measure(y.cpu().numpy(), pred.cpu().numpy(),
                    out['token_logits'].detach().float().cpu().numpy(), L)
        for k,v in m.items(): tok[k] += v

    for k in tok: tok[k] /= len(loader)
    log_metrics(pprint, 'test', None, tok, 'Token')

    pprint(f"✅ 测试完成，共处理样本: {total_samples}")



if __name__ == '__main__':

    # 🧹 Clear GPU memory cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ 已清理 GPU 显存缓存。")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, default=f'{proj_path}/configs/config.yaml')
    args = parser.parse_args()
    config = read_config(args.config)
    _, device, pprint = prepare(config)

    pprint("🚀 正在构建数据加载器 ...")
    train_loader, val_loader, align_valid_loader, test_loader = prepare_dataloaders(config)
    pprint("✅ 数据加载器已就绪。")

    # ========== Stage 0: Build the initial model ==========
    base_model = get_model(pprint, config)
    base_model.glm_lookup = GLMEmbeddingLookup("gLM/batch.pkl.glm.embs.pkl")

    # ========== Stage 1: Align ==========
    pprint("Stage 1: Aligning ...")
    ckpt_align = align(base_model, train_loader, align_valid_loader, config)

    # ========== Stage 2: Fuse ==========
    pprint("Stage 2: Fusing ...")
    ckpt_fuse = fuse(ckpt_align, train_loader, val_loader, config)

    # ========== Stage 2.5: Run post-fusion meta-learning on the best fusion model ==========
    # Create a new model instance and load the best fusion weights
    meta_model = get_model(pprint, config)
    meta_model.glm_lookup = GLMEmbeddingLookup("gLM/batch.pkl.glm.embs.pkl")
    state_fuse = torch.load(ckpt_fuse, map_location='cpu')
    meta_model.load_state_dict(state_fuse)
    meta_model.to(device)

    pprint("Stage 2.5: Meta-learning on fused model ...")
    ckpt_meta = meta_train_post_fusion_maml(
        meta_model, train_loader, config, device, pprint,
        inner_steps=1,
        beta=0.5,
        bce_weight=1.0,
        align_weight=0.0,
        ortho_weight=0.0
    )

    # # 🔐 Save the checkpoint after meta-learning
    # os.makedirs(config.train.save_path, exist_ok=True)
    # ckpt_meta = os.path.join(
    #     config.train.save_path,
    #     f"best_model_fuse_meta_{config.meta.meta_epochs}ep.pt"
    # )
    # torch.save(meta_model.state_dict(), ckpt_meta, _use_new_zipfile_serialization=False)
    # pprint(f"💾 Saving meta-learned fused model -> {ckpt_meta}")

    # # ========== Stage 4: Test using the meta-trained version ==========
    # pprint("Stage 4: Testing (Meta) ...")
    # test(meta_model, ckpt_meta, test_loader, config)

    pprint("train_gLM_true.py版本0.9训练结束")
