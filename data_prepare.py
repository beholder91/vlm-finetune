#!/usr/bin/env python3
# data_prepare.py — 数据集预处理脚本，将数据封装为HuggingFace Dataset格式

import os
import io
import json
import random
import numpy as np
from PIL import Image
import torch
import fitz  # PyMuPDF
from datasets import load_dataset, Dataset, Features, Value, Image as DsImage
from tqdm import tqdm

# 配置参数 - 可以直接在此修改
LOCAL_DATASET_PATH = "./olmOCR-mix-0225/train-s2pdf.parquet"
PDF_DIR = "./olmOCR-mix-0225/pdfs"
OUTPUT_DIR = "./processed_data"
OUTPUT_DATASET_PATH = os.path.join(OUTPUT_DIR, "ocr_dataset")
MAX_SAMPLES = None  # 修改为None可处理全部样本
MAX_SIDE = 1024  # 图像最大边长

def process_image(pdf_id, rotation_prob=0.15, max_side=MAX_SIDE):
    """处理单个图像，返回处理后的PIL图像对象"""
    try:
        # 从本地读取PDF (每个PDF就是单页)
        pdf_path = os.path.join(PDF_DIR, f"{pdf_id}.pdf")
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc[0]  # 只有一页，直接取第一页
        
        # 使用高质量设置渲染PDF
        zoom = 300 / 72  # 300 DPI
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        # 随机旋转
        if random.random() < rotation_prob:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, expand=True)
        
        # Resize: 最长边为 max_side
        w, h = img.size
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        
        # 返回处理后的PIL图像对象
        return {
            "success": True,
            "image": img,
            "width": img.width,
            "height": img.height
        }
    
    except Exception as e:
        print(f"处理图像时出错: {e}, PDF ID: {pdf_id}")
        return {"success": False, "error": str(e)}

def prepare_dataset():
    """将数据集处理并保存为HuggingFace Dataset格式"""
    print(f"加载本地数据集: {LOCAL_DATASET_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载原始数据集
    ds = load_dataset("parquet", data_files=LOCAL_DATASET_PATH)["train"]
    total = len(ds) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(ds))
    print(f"共有 {total} 个样本需要处理")
    
    # 准备新数据集的数据
    dataset_dict = {
        "id": [],
        "image": [],
        "response": [],
        "width": [],
        "height": []
    }
    
    # 处理样本
    processed_count = 0
    error_count = 0
    
    for i, example in enumerate(tqdm(ds, total=total)):
        if MAX_SAMPLES is not None and i >= MAX_SAMPLES:
            break
            
        try:
            # 处理图像
            result = process_image(example["id"])
            
            if result["success"]:
                # 添加到数据集
                dataset_dict["id"].append(example["id"])
                dataset_dict["image"].append(result["image"])
                dataset_dict["response"].append(example["response"])
                dataset_dict["width"].append(result["width"])
                dataset_dict["height"].append(result["height"])
                processed_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            error_count += 1
    
    # 创建HuggingFace Dataset
    print(f"创建HuggingFace Dataset，包含 {len(dataset_dict['id'])} 个样本...")
    features = Features({
        "id": Value("string"),
        "image": DsImage(),
        "response": Value("string"),
        "width": Value("int32"),
        "height": Value("int32")
    })
    
    dataset = Dataset.from_dict(dataset_dict, features=features)
    
    # 保存数据集
    print(f"保存数据集到 {OUTPUT_DATASET_PATH}...")
    dataset.save_to_disk(OUTPUT_DATASET_PATH)
    
    # 保存一个描述文件，方便训练脚本了解数据集结构
    with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), "w") as f:
        json.dump({
            "total_samples": processed_count,
            "failed_samples": error_count,
            "dataset_path": OUTPUT_DATASET_PATH,
            "image_size": f"变化大小，最大边长为 {MAX_SIDE} 像素"
        }, f, indent=2, ensure_ascii=False)
    
    print(f"数据集处理完成. 成功: {processed_count}, 失败: {error_count}")
    print(f"数据集已保存到: {OUTPUT_DATASET_PATH}")
    print(f"您现在可以在训练脚本中使用以下代码加载此数据集:")
    print(f"from datasets import load_from_disk")
    print(f"train_dataset = load_from_disk('{OUTPUT_DATASET_PATH}')")
    print(f"# 数据集中的图像已经处理好，可以直接用于训练，无需额外处理")

if __name__ == "__main__":
    prepare_dataset() 