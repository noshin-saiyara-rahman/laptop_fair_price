import pandas as pd
import numpy as np

raw = pd.read_csv("data/raw/laptops_uncleaned.csv")
df = pd.read_csv("data/raw/laptops_cleaned_v1.csv")

# 1. Remove repeated listings (same specs and price, only index/rating differ)
spec = [c for c in df.columns if c not in ["index", "rating"]]
print("duplicates found:", df.duplicated(subset=spec).sum())
df = df.drop_duplicates(subset=spec).reset_index(drop=True)

# 2. Resolution: columns are mixed up, so take larger/smaller explicitly
a, b = df["resolutioin_height"], df["resolutioin_width"]
df["res_w"], df["res_h"] = np.maximum(a, b), np.minimum(a, b)
df["pixels_mp"] = df.res_w * df.res_h / 1e6
df = df.drop(columns=["resolutioin_height", "resolutioin_width"])

# 3. GPU details from the raw file (cleaned index = raw index + 2)
raw["index"] = raw["Unnamed: 0"] + 2
df = df.merge(raw[["index", "graphic_card"]], on="index", how="left")
g = df["graphic_card"].fillna("")
df["gpu_vram_gb"] = g.str.extract(
    r"^(\d+)\s*GB\s+(?!SSD|HDD)")[0].astype(float).fillna(0)
nv = g.str.extract(r"(RTX|GTX)\s*(\d{3,4})")
df["gpu_num"] = pd.to_numeric(nv[1]).fillna(0)
df["gpu_family"] = np.where(nv[0].notna(), nv[0],
                            np.where(df.gpu_type == "Dedicated", "Other dedicated", "Integrated"))

# 4. Drop noisy or redundant columns
df = df.drop(columns=["rating", "index", "graphic_card", "ram", "processor"])

# 5. Group rare categories into "Other"
for col in ["brand_name", "iter", "os", "processor_brand", "gpu_brand"]:
    vc = df[col].value_counts()
    df[col] = df[col].where(df[col].isin(vc[vc >= 15].index), "Other")

# 6. Missing values and target
for c in ["core_num", "threads_num"]:
    df[c] = df[c].fillna(df[c].median())
df["log_price"] = np.log(df["price"])

df.to_csv("data/clean.csv", index=False)
print(df.shape, "| missing:", df.isna().sum().sum())
