# Keep protein IDs and sequences from train.tsv for gLM input preparation

import pandas as pd

# Read train.tsv
df = pd.read_csv("train.tsv", sep="\t")

# Keep only the entry column
df_entry = df[["Entry"]]

# Save one ID per line to uniprot_ids.txt
df_entry.to_csv("uniprot_ids.txt", index=False, header=False)

print("✅ 已生成 uniprot_ids.txt，每行一个蛋白质ID")
