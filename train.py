#!/usr/bin/env python3
# train.py — RolmOCR 训练脚本，结合Accelerate和Trainer

import os
import torch
import time  # 添加time模块
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
MAX_SAMPLES = 256  # 设置为None表示使用全部样本
EPOCHS = 3
BATCH_SIZE = 4  # 每个GPU的批处理大小
LEARNING_RATE = 3e-5
USE_FP16 = True  # 加回FP16设置
LOGGING_STEPS = 1
SAVE_STEPS = 500
NUM_WORKERS = 4  # 数据加载的线程数
GRADIENT_ACCUMULATION_STEPS = 8  # 加回梯度累积步数设置

def main():
    # 1. 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # 明确设置数值，不使用"auto"
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
    

    # 自定义数据整理函数，处理input_text作为输入提示
    def custom_data_collator(features):
        print(f"Inspecting {len(features)} features in this batch.")
        for i, feature in enumerate(features):
            if "image" not in feature:
                print(f"Feature {i} is missing 'image' key. Keys: {feature.keys()}")
            elif feature["image"] is None:
                print(f"Feature {i} has 'image' key but its value is None.")
            # 您也可以打印 feature["image"] 的类型 type(feature["image"])
        
        # 提取图像和文本
        images = [feature["image"] for feature in features]
        input_texts = [feature["input_text"] for feature in features]
        target_texts = [feature["text"] for feature in features]
        
        # 创建以OCR提示作为文本输入的消息格式
        messages = [
            [
                {"role": "user", "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image"}
                ]}
            ] for input_text in input_texts
        ]
        
        # 使用处理器的聊天模板生成格式化的提示
        formatted_inputs = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        # 对输入和目标进行处理
        inputs = processor(
            text=formatted_inputs,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        # 处理目标文本
        with processor.as_target_processor():
            labels = processor(
                text=target_texts,
                padding=True, 
                return_tensors="pt"
            ).input_ids
        
        # 设置标签
        inputs["labels"] = labels
        
        return inputs

    # 5. 使用自定义数据整理函数替代默认数据整理函数
    data_collator = custom_data_collator

    # 6. TrainingArguments 配置
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
        ddp_find_unused_parameters=False
        )

    # 7. Trainer 实例化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 8. 使用accelerator包装训练过程
    if accelerator.is_main_process:
        print(f"Process {accelerator.process_index}: 即将开始训练...")
        # 尝试加载一个批次，检查数据加载速度
        # 注意：Trainer 会处理数据加载器的分布式封装，直接从dataloader迭代可能不直接反映训练情况
        # try:
        #     batch_start = time.time()
        #     # first_batch = next(iter(dataloader)) # 如果dataloader已被prepare，trainer会处理它的分布式版本
        #     print(f"首批数据加载用时: {time.time() - batch_start:.2f}秒")
        # except Exception as e:
        #     print(f"加载首批数据失败: {e}")
            
    try:
        # 通过Trainer进行训练 (所有进程)
        train_start = time.time()
        trainer.train() # 移出 main_process_first
        if accelerator.is_main_process:
            print(f"训练总用时: {time.time() - train_start:.2f}秒")
        
        # 等待所有进程完成训练
        accelerator.wait_for_everyone()
        
        # 保存模型与处理器 (仅主进程)
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