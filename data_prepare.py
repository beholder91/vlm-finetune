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
import gc  # 导入垃圾回收模块
import base64

fitz.TOOLS.mupdf_display_errors(False)

# 配置参数 - 可以直接在此修改
LOCAL_DATASET_PATH = "./olmOCR-mix-0225/train-s2pdf.parquet"
PDF_DIR = "./olmOCR-mix-0225/pdfs"
MAX_SIDE = 1024  # 图像最大边长

# 图像预处理转换
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(pdf_id, rotation_prob=0.15, max_side=MAX_SIDE):
    """处理单个图像，确保所有图像具有相同尺寸"""
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
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 创建一个固定大小的黑色背景图像
        padded_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        
        # 将调整大小后的图像粘贴到中心位置
        paste_x = (max_side - new_w) // 2
        paste_y = (max_side - new_h) // 2
        padded_img.paste(img, (paste_x, paste_y))
        
        # 转换为Tensor并立即释放PIL图像
        tensor = image_transform(padded_img)
        
        # 释放PIL图像
        del img
        del padded_img
        
        # 返回处理后的结果，不再保留PIL图像
        return {
            "success": True,
            "tensor": tensor,
            "width": max_side,
            "height": max_side
        }
    
    except Exception as e:
        print(f"处理图像时出错: {e}, PDF ID: {pdf_id}")
        return {"success": False, "error": str(e)}

class DynamicOCRDataset(Dataset):
    """动态OCR数据集，实时处理图像数据"""
    
    def __init__(self, parquet_path, pdf_dir, max_samples=None, max_side=MAX_SIDE):
        """初始化数据集
        
        Args:
            parquet_path: parquet文件路径
            pdf_dir: PDF文件目录
            max_samples: 最大样本数，None表示使用全部数据
            max_side: 图像最大边长
        """
        self.pdf_dir = pdf_dir
        self.max_side = max_side
        
        # 加载原始数据集
        self.ds = load_dataset("parquet", data_files=parquet_path)["train"]
        if max_samples is not None:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        
        print(f"加载数据集: {parquet_path}，共{len(self.ds)}个样本")
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        # 获取样本
        example = self.ds[idx]
        
        # 处理图像
        result = process_image(example["id"])
        
        if result["success"]:
            return {
                "image": result["tensor"],
                "text": example["response"],
                "id": example["id"],
                "width": result["width"],
                "height": result["height"]
            }
        else:
            # 处理失败时，提供一个空白图像，与成功处理的图像大小一致
            return {
                "image": torch.zeros((3, self.max_side, self.max_side)),
                "text": example["response"],
                "id": example["id"],
                "width": self.max_side,
                "height": self.max_side
            }

def create_dataloader(batch_size=8, num_workers=4, shuffle=True, max_samples=None):
    """创建动态数据加载器"""
    dataset = DynamicOCRDataset(
        parquet_path=LOCAL_DATASET_PATH,
        pdf_dir=PDF_DIR,
        max_samples=max_samples,
        max_side=MAX_SIDE
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset

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
        
        # 调整大小，保持比例
        new_w, new_h = img.width, img.height
        
        # 创建一个固定大小的黑色背景图像
        padded_img = Image.new("RGB", (target_longest_dim, target_longest_dim), (0, 0, 0))
        
        # 将调整大小后的图像粘贴到中心位置
        paste_x = (target_longest_dim - new_w) // 2
        paste_y = (target_longest_dim - new_h) // 2
        padded_img.paste(img, (paste_x, paste_y))
        
        # 保存为PNG并转换为Base64
        buffer = io.BytesIO()
        padded_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # 主动调用垃圾回收
        del img
        del padded_img
        gc.collect()
        
        return img_base64
        
    except Exception as e:
        raise RuntimeError(f"PDF转换为Base64 PNG时出错: {e}")

if __name__ == "__main__":
    # 测试动态数据加载
    dataloader, dataset = create_dataloader(batch_size=4, num_workers=2, max_samples=10)
    print(f"创建了动态数据加载器，数据集大小: {len(dataset)}")
    
    # 加载一个批次作为示例
    print("加载一个批次数据...")
    for batch in dataloader:
        print(f"批次大小: {len(batch['image'])}")
        print(f"图像形状: {batch['image'][0].shape}")
        print(f"文本示例: {batch['text'][0][:50]}...")
        break 