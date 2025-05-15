import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# 从 data_prepare.py 引入或在此处重新定义这些常量
# 如果这些路径经常变动，考虑使用配置文件或命令行参数
LOCAL_DATASET_PATH = "./olmOCR-mix-0225/train-s2pdf.parquet"
PDF_DIR = "./olmOCR-mix-0225/pdfs"
OUTPUT_CLEANED_DATASET_PATH = "./olmOCR-mix-0225/cleaned_train_data.parquet"

def check_pdf_exists(pdf_id, pdf_dir):
    """检查对应的PDF文件是否存在"""
    if not pdf_id:
        return False
    pdf_path = os.path.join(pdf_dir, f"{pdf_id}.pdf")
    return os.path.exists(pdf_path)

def validate_response(response_str):
    """
    验证response字段：
    1. 是否为非空字符串
    2. 是否可以解析为JSON
    3. 解析后的JSON是否包含非空的 'natural_text' 字段
    """
    if not response_str or not isinstance(response_str, str):
        return False
    try:
        response_json = json.loads(response_str)
        natural_text = response_json.get("natural_text")
        if natural_text and isinstance(natural_text, str) and natural_text.strip():
            return True
    except json.JSONDecodeError:
        return False
    return False

def main():
    print(f"开始加载数据集: {LOCAL_DATASET_PATH}")
    try:
        dataset = load_dataset("parquet", data_files=LOCAL_DATASET_PATH, split="train")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    original_count = len(dataset)
    print(f"原始数据集包含 {original_count} 条样本。")

    if original_count == 0:
        print("数据为空，无需处理。")
        return

    cleaned_data = []
    
    print("开始数据清洗过程...")
    for example in tqdm(dataset, desc="数据清洗进度"):
        pdf_id = example.get("id")
        response_str = example.get("response")

        # 条件1: id 字段存在且对应 PDF 文件存在
        if not pdf_id:
            # print(f"跳过: 'id' 字段缺失。样本内容: {example}") # 可选的详细日志
            continue
        if not check_pdf_exists(pdf_id, PDF_DIR):
            # print(f"跳过: PDF文件不存在。pdf_id: {pdf_id}") # 可选的详细日志
            continue

        # 条件2: response 字段可 JSON 解析，且含 natural_text 非空
        if not validate_response(response_str):
            # print(f"跳过: 'response' 字段无效。pdf_id: {pdf_id}, response: {response_str[:100]}...") # 可选的详细日志
            continue
            
        # 如果所有条件都满足，则保留该样本
        # 为了确保输出的Parquet文件包含所有原始列（如果需要），或者只选择需要的列
        # 这里我们假设保留原始example的所有字段
        cleaned_data.append(example)

    cleaned_count = len(cleaned_data)
    print(f"数据清洗完成。")
    print(f"原始样本数: {original_count}")
    print(f"有效样本数: {cleaned_count}")
    print(f"已移除样本数: {original_count - cleaned_count}")

    if cleaned_count > 0:
        # 将清洗后的数据转换为Pandas DataFrame以便保存为Parquet
        df_cleaned = pd.DataFrame(cleaned_data)
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(OUTPUT_CLEANED_DATASET_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        try:
            df_cleaned.to_parquet(OUTPUT_CLEANED_DATASET_PATH, index=False)
            print(f"清洗后的数据已保存到: {OUTPUT_CLEANED_DATASET_PATH}")
        except Exception as e:
            print(f"保存清洗后的数据失败: {e}")
    else:
        print("没有有效的样本可以保存。")

if __name__ == "__main__":
    main() 