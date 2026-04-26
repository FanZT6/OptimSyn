import pandas as pd

df = pd.read_parquet("/mnt/rl/zz/SynAgent/code/ReflactionSyn/verl/data/book_rubrics/train_1.0.parquet")

# 取 10% 样本
df_sample = df.sample(frac=0.3, random_state=42)

# 保存到新文件
df_sample.to_parquet("/mnt/rl/zz/SynAgent/code/ReflactionSyn/verl/data/book_rubrics/train_0.3.parquet", index=False)
