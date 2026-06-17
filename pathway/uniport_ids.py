# 只保留train.tsv中的蛋白质id和seq去实现glm的输入

import pandas as pd

# 读入 train.tsv
df = pd.read_csv("train.tsv", sep="\t")

# 只保留 entry 列
df_entry = df[["Entry"]]

# 保存为 uniprot_ids.txt，每行一个 ID
df_entry.to_csv("uniprot_ids.txt", index=False, header=False)

print("✅ 已生成 uniprot_ids.txt，每行一个蛋白质ID")
