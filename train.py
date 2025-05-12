#!/usr/bin/env python3
# train.py — RolmOCR 训练脚本，结合Accelerate和Trainer

import os
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers.integrations import WandbCallback
import wandb
from accelerate import Accelerator

# 导入数据处理模块
from data_prepare import create_dataloader

# 训练参数 - 可直接修改
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = "./rolmocr_output"
WANDB_PROJECT = "RolmOCR-finetune"
MAX_SAMPLES = None  # 设置为None表示使用全部样本
EPOCHS = 3
BATCH_SIZE = 1  # 每个GPU的批处理大小
LEARNING_RATE = 3e-5
USE_FP16 = True
LOGGING_STEPS = 100
SAVE_STEPS = 500
NUM_WORKERS = 4  # 数据加载的线程数
GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积步数，减少内存需求

def main():
    # 1. 初始化 Accelerator (不需要DeepSpeed配置，将通过accelerate config处理)
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        log_with="wandb"
    )
    
    # 2. 在主进程中设置 wandb
    if accelerator.is_main_process:
        wandb.init(project=WANDB_PROJECT)
        accelerator.init_trackers(WANDB_PROJECT)
        print("W&B 初始化完成")

    # 3. 创建动态数据加载器
    if accelerator.is_main_process:
        print("创建动态数据加载器...")
    dataloader, train_dataset = create_dataloader(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        max_samples=MAX_SAMPLES
    )
    
    if accelerator.is_main_process:
        print(f"成功创建数据集，包含 {len(train_dataset)} 个样本")
    
    if len(train_dataset) == 0:
        if accelerator.is_main_process:
            print("错误: 没有加载到任何样本，请检查数据集")
        return
    
    # 4. 加载处理器与模型
    if accelerator.is_main_process:
        print(f"加载模型和处理器: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 5. 数据 collator
    data_collator = DataCollatorWithPadding(processor, pad_to_multiple_of=8)

    # 6. TrainingArguments 配置 - 不需要指定DeepSpeed配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        fp16=USE_FP16,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        report_to=["wandb"],
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        # 不需要指定DeepSpeed配置，将由accelerate处理
        ddp_find_unused_parameters=False,
    )

    # 7. Trainer 实例化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[WandbCallback]
    )

    # 8. 使用accelerator包装训练过程
    with accelerator.main_process_first():
        if accelerator.is_main_process:
            print("开始训练...")
        try:
            # 通过Trainer进行训练
            trainer.train()
            
            # 保存模型与处理器
            if accelerator.is_main_process:
                trainer.save_model(OUTPUT_DIR)
                processor.save_pretrained(OUTPUT_DIR)
                print(f"训练完成，模型保存在 {OUTPUT_DIR}")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"训练过程中出错: {e}")
    
    # 9. 结束跟踪
    accelerator.end_training()

if __name__ == "__main__":
    main()