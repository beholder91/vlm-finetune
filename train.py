#!/usr/bin/env python3
# train.py — RolmOCR 训练脚本，直接加载预处理好的数据集进行训练

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers.integrations import WandbCallback, TensorBoardCallback
import wandb
from datasets import load_from_disk

# 训练参数 - 可直接修改
MODEL_ID = "Qwen/Qwen2.5-VL-2B-Instruct"
DATASET_PATH = "./processed_data/ocr_dataset"  # 预处理好的数据集路径
OUTPUT_DIR = "./rolmocr_output"
WANDB_PROJECT = "RolmOCR-finetune"
MAX_SAMPLES = None  # 设置为None表示使用全部样本
EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 3e-5
USE_FP16 = True
LOGGING_STEPS = 100
EVAL_STEPS = 500
SAVE_STEPS = 500
NUM_WORKERS = 4  # 数据加载的线程数

# 图像预处理转换
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PreprocessedOCRDataset(Dataset):
    """包装预处理好的HuggingFace数据集，便于训练使用"""
    
    def __init__(self, dataset, transform=None):
        """初始化数据集
        
        Args:
            dataset: HuggingFace Dataset对象
            transform: 图像转换函数
        """
        self.dataset = dataset
        self.transform = transform
        
        print(f"加载了 {len(dataset)} 个预处理好的样本")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # 获取图像并应用转换
        img = sample["image"]
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        
        return {
            "img_tensor": img_tensor,
            "labels": sample["response"],
            "id": sample["id"]
        }

def main():
    # 1. 环境变量与 W&B 初始化
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    wandb.init()

    # 2. 直接加载预处理好的数据集
    print(f"正在加载预处理好的数据集: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print(f"错误: 数据集目录 {DATASET_PATH} 不存在，请先运行 data_prepare.py")
        return
    
    try:
        # 加载预处理好的HuggingFace数据集
        hf_dataset = load_from_disk(DATASET_PATH)
        
        # 如果需要限制样本数
        if MAX_SAMPLES is not None and MAX_SAMPLES < len(hf_dataset):
            hf_dataset = hf_dataset.select(range(MAX_SAMPLES))
        
        # 包装为PyTorch Dataset
        train_dataset = PreprocessedOCRDataset(
            dataset=hf_dataset,
            transform=image_transform
        )
        
    except Exception as e:
        print(f"加载数据集出错: {e}")
        return
    
    if len(train_dataset) == 0:
        print("错误: 没有加载到任何样本，请检查数据集")
        return
    
    # 3. 加载分词器与模型
    print(f"加载模型和分词器: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 4. 数据 collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # 5. TrainingArguments 配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        fp16=USE_FP16,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        report_to=["wandb", "tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="loss",
        dataloader_num_workers=NUM_WORKERS,  # 多线程加载数据
    )

    # 6. Trainer 实例化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[WandbCallback, TensorBoardCallback]
    )

    # 7. 开始训练
    print("开始训练...")
    try:
        trainer.train()
        
        # 8. 保存模型与 Tokenizer
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"训练完成，模型保存在 {OUTPUT_DIR}")
    except Exception as e:
        print(f"训练过程中出错: {e}")

if __name__ == "__main__":
    main()
