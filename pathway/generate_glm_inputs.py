#!/usr/bin/env python3
# 根据构造的glm_input.tsv进一步调整结果，满足最终的glm输入需要

"""
make_glm_inputs_from_files.py

作用：
  1) 从 glm_input.tsv（contig_id pathway_id proteins）生成无表头的 contig_to_prots 文件：
       glm_input_contig2prots.tsv  (contig_id \t prot1;prot2;...)
  2) 从 train.tsv (Entry, Sequence) 提取对应蛋白序列并生成 proteins_old.fa（去重、去*）
  3) 生成 missing_proteins.txt 列出在 glm_input 中出现但在 train.tsv 找不到序列的蛋白 ID

用法：
  python make_glm_inputs_from_files.py \
      --glm glm_input.tsv \
      --train train.tsv \
      --outdir results_glm_input

输出：
  results_glm_input/glm_input_contig2prots.tsv
  results_glm_input/proteins_old.fa
  results_glm_input/missing_proteins.txt
"""

import argparse
import os
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--glm", default="glm_input.tsv", help="输入 glm_input.tsv 文件 (contig_id pathway_id proteins)")
    p.add_argument("--train", default="train.tsv", help="train.tsv，至少包含 Entry 和 Sequence 两列")
    p.add_argument("--outdir", default="results_glm_input", help="输出目录")
    return p.parse_args()

def read_train_sequences(train_path):
    """
    读取 train.tsv，返回 dict: entry -> sequence
    尝试识别列名（大小写容错）。
    如果无法用 pandas 读取（分隔符问题），做简单的手工解析（首两列为 id, seq）。
    """
    import pandas as pd
    # 尝试用制表符读
    try:
        df = pd.read_csv(train_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception:
        # 退而求其次，用任意空白分隔
        df = pd.read_csv(train_path, sep=r"\s+", engine="python", dtype=str, keep_default_na=False)

    cols = [c.lower() for c in df.columns]
    entry_col = None
    seq_col = None
    for c, lc in zip(df.columns, cols):
        if "entry" == lc or lc.endswith("entry") or "uniprot" in lc or "id" == lc:
            entry_col = c
            break
    for c, lc in zip(df.columns, cols):
        if "sequence" == lc or lc.endswith("sequence") or "seq" == lc or "protein" in lc:
            seq_col = c
            break

    # 如果没识别到，尝试第一两列
    if entry_col is None or seq_col is None:
        if df.shape[1] >= 2:
            entry_col = df.columns[0]
            seq_col = df.columns[1]
        else:
            raise ValueError("无法识别 train.tsv 中的 Entry/Sequence 列，请确保至少包含两列：Entry 与 Sequence。")

    seq_dict = {}
    for _, r in df.iterrows():
        eid = str(r[entry_col]).strip()
        seq = str(r[seq_col]).strip().replace(" ", "").replace("*", "")
        if eid:
            seq_dict[eid] = seq
    return seq_dict

def process_glm_input(glm_path, out_contig_path):
    """
    读取 glm_input.tsv，跳过 header（若第一行包含 'contig' 和 'proteins' 则视作 header）
    每行取第一列为 contig_id，最后一列为 proteins 字符串（保留 +/- 前缀在输出，但 proteins_old.fa 使用无前缀 id）
    输出 contig_to_prots.tsv（无 header）
    并返回 proteins_id_set（去除方向符号）
    """
    proteins_set = set()
    lines_out = []

    with open(glm_path, "r", encoding="utf-8") as fin:
        all_lines = [ln.rstrip("\n") for ln in fin]

    if not all_lines:
        raise ValueError(f"{glm_path} 是空文件")

    # 判断是否有 header
    first = all_lines[0].strip().lower()
    start_idx = 0
    if ("contig" in first and "protein" in first) or ("proteins" in first and "contig" in first):
        start_idx = 1

    for ln in all_lines[start_idx:]:
        if not ln.strip():
            continue
        # 先尝试按 tab 切分，如果只有一列再用空白分割
        parts = ln.split("\t")
        if len(parts) < 2:
            parts = ln.split()
            if len(parts) < 2:
                # 无法解析跳过
                continue
        contig_id = parts[0].strip()
        proteins_field = parts[-1].strip()  # 最后一列应该是 proteins
        # 规范化：各蛋白以;分隔
        prots = [p.strip() for p in proteins_field.split(";") if p.strip()]
        # 记录无方向的 id
        for p in prots:
            pid = p[1:] if (p.startswith("+") or p.startswith("-")) else p
            proteins_set.add(pid)
        lines_out.append((contig_id, ";".join(prots)))

    # 写 contig_to_prots.tsv（无表头）
    with open(out_contig_path, "w", encoding="utf-8") as fout:
        for contig_id, protstr in lines_out:
            fout.write(f"{contig_id}\t{protstr}\n")

    return proteins_set

def write_proteins_fa(proteins_set, seq_dict, out_fa_path, missing_path):
    """
    proteins_set: set of protein ids (no +/-)
    seq_dict: mapping entry->sequence from train.tsv
    写 proteins_old.fa（按 sorted order），并记录未找到的 id 到 missing_path
    """
    missing = []
    found_count = 0
    with open(out_fa_path, "w", encoding="utf-8") as fout:
        for pid in sorted(proteins_set):
            seq = seq_dict.get(pid)
            if seq and seq.strip():
                # 去掉 '*' 并去空格
                seq_clean = seq.replace("*", "").replace(" ", "").strip()
                fout.write(f">{pid}\n{seq_clean}\n")
                found_count += 1
            else:
                missing.append(pid)

    with open(missing_path, "w", encoding="utf-8") as fmiss:
        for m in missing:
            fmiss.write(m + "\n")

    return found_count, len(missing)

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    glm_path = Path(args.glm)
    train_path = Path(args.train)
    out_contig_path = outdir / "glm_input_contig2prots.tsv"
    out_fa_path = outdir / "proteins_old.fa"
    missing_path = outdir / "missing_proteins.txt"

    print("[1/3] 读取 train 序列...")
    seq_dict = read_train_sequences(train_path)
    print(f"  在 train.tsv 中检测到 {len(seq_dict)} 条 entry->sequence 映射")

    print("[2/3] 处理 glm_input，生成 contig->proteins（去掉 pathway, 跳过 header）...")
    proteins_set = process_glm_input(glm_path, out_contig_path)
    print(f"  从 glm_input 提取到 {len(proteins_set)} 个唯一的 protein IDs")

    print("[3/3] 生成 proteins_old.fa（并记录缺失）...")
    found, missing = write_proteins_fa(proteins_set, seq_dict, out_fa_path, missing_path)
    print(f"  proteins_old.fa 写入 {found} 条序列")
    if missing:
        print(f"  WARNING: 有 {missing} 个 protein 在 {train_path} 中未找到序列，已写入 {missing_path}")
    else:
        print("  所有 protein 都在 train.tsv 中找到序列")

    print("\n完成。输出文件：")
    print("  -", out_contig_path)
    print("  -", out_fa_path)
    print("  -", missing_path)

if __name__ == "__main__":
    main()
