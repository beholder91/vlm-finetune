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
from datasets import load_dataset
from tqdm import tqdm
import pickle

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
    """处理单个图像，返回处理后的PIL图像对象"""
    try:
        # 从本地读取PDF (每个PDF就是单页)
        pdf_path = os.path.join(PDF_DIR, f"{pdf_id}.pdf")
        pdf_doc = fitz.open(pdf_path)
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

class OCRDataset(Dataset):
    """OCR数据集PyTorch实现"""
    
    def __init__(self, samples, transform=None):
        """初始化数据集
        
        Args:
            samples: 样本列表，每个样本是包含image和response的字典
            transform: 图像转换函数
        """
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 获取图像并应用转换
        img = sample["image"]
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        
        return {
            "image": img_tensor,
            "text": sample["response"],
            "id": sample["id"],
            "width": sample["width"],
            "height": sample["height"]
        }

def prepare_dataset():
    """直接将数据集处理并保存为PyTorch Dataset格式"""
    print(f"加载本地数据集: {LOCAL_DATASET_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载原始数据集
    ds = load_dataset("parquet", data_files=LOCAL_DATASET_PATH)["train"]
    if MAX_SAMPLES is not None:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    
    total = len(ds)
    print(f"共有 {total} 个样本需要处理")
    
    # 处理数据并收集有效样本
    processed_samples = []
    error_count = 0
    
    for i, example in enumerate(tqdm(ds, desc="处理样本")):
        try:
            # 处理图像
            result = process_image(example["id"])
            
            if result["success"]:
                processed_samples.append({
                    "id": example["id"],
                    "image": result["image"],
                    "response": example["response"],
                    "width": result["width"],
                    "height": result["height"]
                })
            else:
                error_count += 1
                
        except Exception as e:
            print(f"处理样本时出错: {e}, ID: {example['id']}")
            error_count += 1
            
        # 定期输出进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1}/{total} 个样本，成功: {len(processed_samples)}，失败: {error_count}")
    
    processed_count = len(processed_samples)
    print(f"数据处理完成. 成功: {processed_count}, 失败: {error_count}")
    
    # 创建PyTorch Dataset
    print(f"创建PyTorch Dataset，包含 {processed_count} 个样本...")
    ocr_dataset = OCRDataset(processed_samples, transform=image_transform)
    
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
    print(f"您现在可以在训练脚本中使用以下代码加载此数据集:")
    print(f"import pickle")
    print(f"with open('{OUTPUT_DATASET_PATH}', 'rb') as f:")
    print(f"    train_dataset = pickle.load(f)")
    print(f"# 数据集中的图像已经处理好，可以直接用于训练")

if __name__ == "__main__":
    prepare_dataset() 