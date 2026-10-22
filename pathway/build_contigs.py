# Build glm_input.tsv

import pandas as pd
import random
import argparse

def build_contigs(input_file, output_file, seed=42, min_len=15, max_len=30):
    # Set a fixed random seed
    random.seed(seed)

    # Read protein2pathway.tsv
    # Format: uniprot_id   kegg_gene   pathway_id
    df = pd.read_csv(input_file, sep="\t")

    # Group by pathway_id
    groups = df.groupby("pathway_id")["uniprot_id"].apply(list)

    contigs = []
    contig_id = 0

    for pathway, proteins in groups.items():
        # Deduplicate proteins
        proteins = list(set(proteins))

        # Skip small pathways
        if len(proteins) <= min_len:
            continue

        # Shuffle the order
        random.shuffle(proteins)

        # Split into chunks of at most max_len proteins
        for i in range(0, len(proteins), max_len):
            chunk = proteins[i:i+max_len]
            if len(chunk) < min_len:
                continue  # Discard chunks shorter than min_len

            # Assign random orientation markers
            chunk_with_dir = [random.choice(["+", "-"]) + p for p in chunk]

            contigs.append((f"contig_{contig_id}", pathway, ";".join(chunk_with_dir)))
            contig_id += 1

    # Save the results
    with open(output_file, "w") as f:
        f.write("contig_id\tpathway_id\tproteins\n")
        for cid, pathway, seq in contigs:
            f.write(f"{cid}\t{pathway}\t{seq}\n")

    print(f"✅ 共生成 {len(contigs)} 条 contigs，保存到 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按 pathway 拼接 contigs")
    parser.add_argument("-i", "--input", default="protein2pathway.tsv", help="输入文件 (protein2pathway.tsv)")
    parser.add_argument("-o", "--output", default="glm_input.tsv", help="输出文件 (glm_input.tsv)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="随机种子 (默认=42)")
    parser.add_argument("--min_len", type=int, default=15, help="contig 最小长度 (默认=15)")
    parser.add_argument("--max_len", type=int, default=30, help="contig 最大长度 (默认=30)")
    args = parser.parse_args()

    build_contigs(args.input, args.output, args.seed, args.min_len, args.max_len)
