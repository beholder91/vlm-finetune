#!/usr/bin/env python3
# train.py — RolmOCR 训练脚本，从预处理数据加载并训练模型

import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers.integrations import WandbCallback, TensorBoardCallback
import wandb

# 训练参数 - 可直接修改
MODEL_ID = "Qwen/Qwen2.5-VL-2B-Instruct"
DATA_DIR = "./processed_data"
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

class ProcessedOCRDataset(Dataset):
    """处理好的OCR数据集类"""
    
    def __init__(self, metadata_file, max_samples=None):
        """初始化数据集
        
        Args:
            metadata_file: JSONL格式的元数据文件路径
            max_samples: 最大样本数，None表示加载全部
        """
        self.samples = []
        
        print(f"从 {metadata_file} 加载元数据...")
        with open(metadata_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                    
                metadata = json.loads(line.strip())
                self.samples.append(metadata)
        
        print(f"加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像张量
        img_tensor = torch.load(sample["img_path"])
        
        return {
            "img_tensor": img_tensor,
            "labels": sample["response"],
            "id": sample["id"]
        }

def main():
    # 1. 环境变量与 W&B 初始化
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    wandb.init()

    # 2. 加载数据集
    metadata_file = os.path.join(DATA_DIR, "metadata.jsonl")
    if not os.path.exists(metadata_file):
        print(f"错误: 元数据文件 {metadata_file} 不存在，请先运行 data_prepare.py")
        return
        
    train_dataset = ProcessedOCRDataset(metadata_file, max_samples=MAX_SAMPLES)
    
    if len(train_dataset) == 0:
        print("错误: 没有加载到任何样本，请检查预处理数据")
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
