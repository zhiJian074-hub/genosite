# 根据uniprot_ids生成protein2pathway.tsv
# 完成对于pathway的映射分类

import requests
import time

# ========== 通用请求函数 ==========
def safe_get(url, retries=3, wait=3):
    """
    带重试机制的 GET 请求
    """
    for i in range(retries):
        try:
            r = requests.get(url, timeout=20)  # 20 秒超时
            if r.status_code == 200 and r.text.strip():
                return r
        except Exception as e:
            print(f"⚠️ 请求失败: {e}，第 {i+1}/{retries} 次重试...")
            time.sleep(wait)
    return None

# ========== UniProt -> KEGG Gene ==========
def uniprot_to_kegg(uniprot_id):
    url = f"http://rest.kegg.jp/conv/genes/uniprot:{uniprot_id}"
    r = safe_get(url)
    if not r:
        return None
    try:
        return r.text.strip().split("\t")[1]  # 例如 "hsa:1956"
    except:
        return None

# ========== KEGG Gene -> Pathways ==========
def kegg_gene_to_pathways(kegg_gene):
    url = f"http://rest.kegg.jp/link/pathway/{kegg_gene}"
    r = safe_get(url)
    if not r:
        return []
    try:
        pathways = [line.split("\t")[1].replace("path:", "") for line in r.text.strip().split("\n")]
        return pathways
    except:
        return []

# ========== 主流程 ==========
def process_uniprot_list(input_file, output_file):
    with open(input_file, "r") as f:
        uniprot_ids = [line.strip() for line in f if line.strip()]

    with open(output_file, "w") as out:
        out.write("uniprot_id\tkegg_gene\tpathway_id\n")

        for idx, uid in enumerate(uniprot_ids, 1):
            kegg_gene = uniprot_to_kegg(uid)
            if not kegg_gene:
                print(f"❌ {uid} 无法映射到 KEGG")
                continue

            pathways = kegg_gene_to_pathways(kegg_gene)
            if not pathways:
                print(f"⚠️ {uid} ({kegg_gene}) 没有关联通路")
                continue

            for pw in pathways:
                out.write(f"{uid}\t{kegg_gene}\t{pw}\n")

            print(f"✅ {idx}/{len(uniprot_ids)}: {uid} ({kegg_gene}) → {len(pathways)} 条通路")

            time.sleep(0.5)  # 避免请求过快被 KEGG 屏蔽

    print(f"\n全部完成 ✅，结果保存在 {output_file}")

# ========== 入口 ==========
if __name__ == "__main__":
    input_file = "uniprot_ids.txt"       # 输入：每行一个 UniProt ID
    output_file = "protein2pathway.tsv"  # 输出：映射结果
    process_uniprot_list(input_file, output_file)
