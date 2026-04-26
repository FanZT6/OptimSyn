import pandas as pd
import os

file_path = os.getenv("MMLU_PRO_PATH", "<YOUR_MMLU_PRO_PARQUET_PATH>")

# 读取
df = pd.read_parquet(file_path)

# 查看前 1 条
print(df.head(1))

# 如果你只想看 JSON 结构，可以转成字典
print(df.iloc[200].to_dict())
