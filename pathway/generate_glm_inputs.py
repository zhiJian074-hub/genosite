#!/usr/bin/env python3
# Convert the constructed glm_input.tsv into the files required by gLM

"""
make_glm_inputs_from_files.py

Purpose:
  1) Create a headerless contig-to-proteins file from glm_input.tsv (contig_id, pathway_id, proteins):
       glm_input_contig2prots.tsv  (contig_id \t prot1;prot2;...)
  2) Extract protein sequences from train.tsv (Entry, Sequence) and create proteins_old.fa (deduplicated, with '*' removed)
  3) Create missing_proteins.txt for protein IDs present in glm_input but absent from train.tsv

Usage:
  python make_glm_inputs_from_files.py \
      --glm glm_input.tsv \
      --train train.tsv \
      --outdir results_glm_input

Outputs:
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
    Read train.tsv and return a dictionary mapping entry to sequence.
    Detect column names without regard to case.
    If pandas cannot parse the delimiters, fall back to simple parsing with ID and sequence in the first two columns.
    """
    import pandas as pd
    # Try tab-separated input
    try:
        df = pd.read_csv(train_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception:
        # Fall back to splitting on whitespace
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

    # If columns are not recognized, try the first two
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
    Read glm_input.tsv and skip a header if the first row contains both 'contig' and 'proteins'.
    Use the first column as contig_id and the last as proteins; retain +/- prefixes in output but remove them for proteins_old.fa.
    Write contig_to_prots.tsv without a header.
    Return proteins_id_set with orientation markers removed.
    """
    proteins_set = set()
    lines_out = []

    with open(glm_path, "r", encoding="utf-8") as fin:
        all_lines = [ln.rstrip("\n") for ln in fin]

    if not all_lines:
        raise ValueError(f"{glm_path} 是空文件")

    # Detect a header
    first = all_lines[0].strip().lower()
    start_idx = 0
    if ("contig" in first and "protein" in first) or ("proteins" in first and "contig" in first):
        start_idx = 1

    for ln in all_lines[start_idx:]:
        if not ln.strip():
            continue
        # Split on tabs first, then on whitespace if only one column is found
        parts = ln.split("\t")
        if len(parts) < 2:
            parts = ln.split()
            if len(parts) < 2:
                # Skip rows that cannot be parsed
                continue
        contig_id = parts[0].strip()
        proteins_field = parts[-1].strip()  # The final column should contain proteins
        # Normalize proteins as semicolon-separated values
        prots = [p.strip() for p in proteins_field.split(";") if p.strip()]
        # Record IDs without orientation markers
        for p in prots:
            pid = p[1:] if (p.startswith("+") or p.startswith("-")) else p
            proteins_set.add(pid)
        lines_out.append((contig_id, ";".join(prots)))

    # Write contig_to_prots.tsv without a header
    with open(out_contig_path, "w", encoding="utf-8") as fout:
        for contig_id, protstr in lines_out:
            fout.write(f"{contig_id}\t{protstr}\n")

    return proteins_set

def write_proteins_fa(proteins_set, seq_dict, out_fa_path, missing_path):
    """
    proteins_set: set of protein ids (no +/-)
    seq_dict: mapping entry->sequence from train.tsv
    Write proteins_old.fa in sorted order and record missing IDs in missing_path.
    """
    missing = []
    found_count = 0
    with open(out_fa_path, "w", encoding="utf-8") as fout:
        for pid in sorted(proteins_set):
            seq = seq_dict.get(pid)
            if seq and seq.strip():
                # Remove '*' and whitespace
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
