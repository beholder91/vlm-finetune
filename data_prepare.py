#!/usr/bin/env python3
# data_prepare.py — 数据集预处理脚本

import os
import io
import json
import random
from PIL import Image
import torch
import fitz  # PyMuPDF
from datasets import load_dataset
from tqdm import tqdm

# 配置参数 - 可以直接在此修改
LOCAL_DATASET_PATH = "./olmOCR-mix-0225/train-s2pdf.parquet"
PDF_DIR = "./olmOCR-mix-0225/pdfs"
OUTPUT_DIR = "./processed_data"
MAX_SAMPLES = 100  # 修改为None可处理全部样本

def process_image(pdf_id, rotation_prob=0.15, max_side=1024):
    """处理单个图像，返回处理后的张量"""
    try:
        # 从本地读取PDF (每个PDF就是单页)
        pdf_path = os.path.join(PDF_DIR, f"{pdf_id}.pdf")
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc[0]  # 只有一页，直接取第一页
        pix = page.get_pixmap()
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        # 随机旋转
        if random.random() < rotation_prob:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, expand=True)
        
        # Resize: 最长边为 max_side
        w, h = img.size
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
        
        # 获取新尺寸并创建张量
        new_w, new_h = img.size
        img_tensor = torch.tensor(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(new_h, new_w, 3)
             .permute(2, 0, 1)),
            dtype=torch.float32
        ) / 255.0
        
        return {
            "success": True,
            "tensor": img_tensor,
            "width": new_w,
            "height": new_h
        }
    
    except Exception as e:
        print(f"处理图像时出错: {e}, PDF ID: {pdf_id}")
        return {"success": False, "error": str(e)}

def process_dataset():
    """处理数据集并保存为文件"""
    print(f"加载本地数据集: {LOCAL_DATASET_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据集
    ds = load_dataset("parquet", data_files=LOCAL_DATASET_PATH)["train"]
    total = len(ds) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(ds))
    print(f"共有 {total} 个样本需要处理")
    
    # 处理样本并保存
    processed_count = 0
    error_count = 0
    
    with open(os.path.join(OUTPUT_DIR, "metadata.jsonl"), "w") as f_meta:
        for i, example in enumerate(tqdm(ds, total=total)):
            if MAX_SAMPLES is not None and i >= MAX_SAMPLES:
                break
                
            try:
                # 处理图像，不再需要传递页码
                result = process_image(example["id"])
                
                if result["success"]:
                    # 保存图像张量
                    img_path = os.path.join(OUTPUT_DIR, f"img_{i}.pt")
                    torch.save(result["tensor"], img_path)
                    
                    # 保存元数据
                    metadata = {
                        "id": example["id"],
                        "img_path": img_path,
                        "response": example["response"],
                        "width": result["width"],
                        "height": result["height"]
                    }
                    f_meta.write(json.dumps(metadata) + "\n")
                    processed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                error_count += 1
                
    print(f"数据处理完成. 成功: {processed_count}, 失败: {error_count}")
    print(f"元数据文件保存在: {os.path.join(OUTPUT_DIR, 'metadata.jsonl')}")

if __name__ == "__main__":
    process_dataset() 