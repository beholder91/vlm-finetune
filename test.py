from datasets import load_dataset
import json

def load_local_parquet(path: str):
    # 通过指定"parquet"格式和本地文件路径加载数据集
    ds = load_dataset("parquet", data_files=path)["train"]
    return ds

# 调用示例
dataset = load_local_parquet("./olmOCR-mix-0225/train-s2pdf.parquet")
print(dataset)


# import pandas as pd
# df = pd.read_parquet("./olmOCR-mix-0225/train-s2pdf.parquet")
# print(df.info())
# print(df.isnull().sum()) # 检查每列的空值数量
# # 查找 'id' 列为空的行
# print(df[df['id'].isnull()])
# # 查找 'response' 列为空的行 (如果 response 是一个嵌套结构，检查会更复杂)
# print(df[df['response'].isnull()])