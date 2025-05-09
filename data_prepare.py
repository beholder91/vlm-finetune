import os
import io
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import fitz  # PyMuPDF
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
import pickle
import base64
import gc  # 导入垃圾回收模块

fitz.TOOLS.mupdf_display_errors(False)

# 配置参数 - 可以直接在此修改
LOCAL_DATASET_PATH = "./olmOCR-mix-0225/train-s2pdf.parquet"
PDF_DIR = "./olmOCR-mix-0225/pdfs"
OUTPUT_DIR = "./processed_data"
OUTPUT_DATASET_PATH = os.path.join(OUTPUT_DIR, "ocr_pytorch_dataset.pkl")
MAX_SAMPLES = 1000  # 修改为None可处理全部样本
MAX_SIDE = 1024  # 图像最大边长


# 图像预处理转换
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(pdf_id, rotation_prob=0.15, max_side=MAX_SIDE):
    """处理单个图像，直接返回Tensor和尺寸信息，不返回PIL图像"""
    try:
        # 从本地读取PDF (每个PDF就是单页)
        pdf_path = os.path.join(PDF_DIR, f"{pdf_id}.pdf")
        
        # 使用with语句打开PDF文件，确保自动关闭
        with fitz.open(pdf_path) as pdf_doc:
            page = pdf_doc[0]  # 只有一页，直接取第一页
            
            # 直接将PDF页面转换为PIL图像
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 随机旋转
        if random.random() < rotation_prob:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, expand=True)
        
        # Resize: 最长边为 max_side
        w, h = img.size
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        
        # 转换为Tensor并立即释放PIL图像
        tensor = image_transform(img)
        width, height = img.width, img.height
        
        # 释放PIL图像
        del img
        
        # 返回处理后的结果，不再保留PIL图像
        return {
            "success": True,
            "tensor": tensor,
            "width": width,
            "height": height
        }
    
    except Exception as e:
        print(f"处理图像时出错: {e}, PDF ID: {pdf_id}")
        return {"success": False, "error": str(e)}

class OCRDataset(Dataset):
    """OCR数据集PyTorch实现"""
    
    def __init__(self, samples, transform=None):
        """初始化数据集
        
        Args:
            samples: 样本列表，每个样本是包含tensor和response的字典
            transform: 图像转换函数(此处已经在预处理阶段使用了)
        """
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            "image": sample["tensor"],
            "text": sample["response"],
            "id": sample["id"],
            "width": sample["width"],
            "height": sample["height"]
        }

def process_example(example, process_images=True):
    """单个或批量样本处理函数，用于HF Datasets的map方法"""
    if not process_images:
        return example
        
    result = process_image(example["id"])
    
    if result["success"]:
        return {
            "id": example["id"],
            "response": example["response"],
            "tensor": result["tensor"],
            "width": result["width"],
            "height": result["height"],
            "success": True
        }
    else:
        return {
            "id": example["id"],
            "response": example["response"],
            "success": False,
            "error": result.get("error", "未知错误")
        }

def prepare_dataset():
    """使用流式处理和HF Datasets的map方法处理数据并保存为PyTorch Dataset格式"""
    print(f"加载本地数据集: {LOCAL_DATASET_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载原始数据集
    ds = load_dataset("parquet", data_files=LOCAL_DATASET_PATH)["train"]
    if MAX_SAMPLES is not None:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    
    total = len(ds)
    print(f"共有 {total} 个样本需要处理")
    
    # 使用流式处理和map方法处理数据
    processed_samples = []
    error_count = 0
    batch_size = 32  # 可根据内存情况调整批处理大小
    
    # 流式处理数据集并收集有效样本
    for i in tqdm(range(0, total, batch_size), desc="处理样本批次"):
        end_idx = min(i + batch_size, total)
        batch_ds = ds.select(range(i, end_idx))
        
        # 处理当前批次的所有样本
        processed_batch = [process_example(example) for example in batch_ds]
        
        # 收集成功处理的样本
        for result in processed_batch:
            if result["success"]:
                processed_samples.append({
                    "id": result["id"],
                    "tensor": result["tensor"],
                    "response": result["response"],
                    "width": result["width"],
                    "height": result["height"]
                })
            else:
                error_count += 1
        
        # 手动调用垃圾回收
        gc.collect()
        
        # 每处理10个批次输出一次进度
        if (i // batch_size + 1) % 10 == 0:
            print(f"已处理 {end_idx}/{total} 个样本，成功: {len(processed_samples)}，失败: {error_count}")
    
    processed_count = len(processed_samples)
    print(f"数据处理完成. 成功: {processed_count}, 失败: {error_count}")
    
    # 创建PyTorch Dataset
    print(f"创建PyTorch Dataset，包含 {processed_count} 个样本...")
    ocr_dataset = OCRDataset(processed_samples, transform=None)  # 已经转换为tensor，不需要再次转换
    
    # 保存数据集
    print(f"保存数据集到 {OUTPUT_DATASET_PATH}...")
    with open(OUTPUT_DATASET_PATH, 'wb') as f:
        pickle.dump(ocr_dataset, f)
    
    # 保存一个描述文件，方便训练脚本了解数据集结构
    with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), "w") as f:
        json.dump({
            "total_samples": total,
            "processed_samples": processed_count,
            "failed_samples": error_count,
            "dataset_path": OUTPUT_DATASET_PATH,
            "image_size": f"变化大小，最大边长为 {MAX_SIDE} 像素",
            "pdf_converter": "PyMuPDF"
        }, f, indent=2, ensure_ascii=False)
    
    print(f"数据集已保存到: {OUTPUT_DATASET_PATH}")

def render_pdf_to_base64png(pdf_path: str, target_longest_dim: int = 2048) -> str:
    try:
        # 使用with语句打开PDF文件
        with fitz.open(pdf_path) as pdf_doc:
            page = pdf_doc[0]
            
            # 获取页面尺寸
            rect = page.rect
            width, height = rect.width, rect.height
            longest_dim = max(width, height)
            
            # 计算缩放比例
            scale = target_longest_dim / longest_dim
            
            # 设置渲染参数
            matrix = fitz.Matrix(scale, scale)
            
            # 渲染为像素图
            pix = page.get_pixmap(matrix=matrix)
            
            # 转换为PIL图像
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 保存为PNG并转换为Base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # 主动调用垃圾回收
        del img
        gc.collect()
        
        return img_base64
        
    except Exception as e:
        raise RuntimeError(f"PDF转换为Base64 PNG时出错: {e}")

if __name__ == "__main__":
    prepare_dataset() 