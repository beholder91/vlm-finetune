#!/usr/bin/env python3
# train.py — RolmOCR 训练脚本，结合Accelerate和Trainer

import os
import torch
import time
import traceback
from PIL import Image
import io
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
NUM_WORKERS = 0  # 数据加载的线程数
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
        # print(f"[Collator PID {os.getpid()}] Received {len(features)} features for collation.")
        
        # 过滤掉在 __getitem__ 中标记为包含错误的样本
        valid_features = []
        for i, feature in enumerate(features):
            if feature.get("__error__") is not None:
                # print(f"[Collator PID {os.getpid()}] Skipping feature {i} (ID: {feature.get('id', 'UnknownID')}) due to __error__: {feature['__error__']}")
                continue
            if feature.get("image") is None:
                # print(f"[Collator PID {os.getpid()}] Skipping feature {i} (ID: {feature.get('id', 'UnknownID')}) because image is None.")
                continue
            valid_features.append(feature)

        if not valid_features:
            # print(f"[Collator PID {os.getpid()}] No valid features left after filtering errors. Original count: {len(features)}.")
            # 根据 Trainer 的行为，返回一个空字典或特定的结构可能会导致后续错误，但至少 collate 本身不崩
            # 或者，如果严格要求，可以抛出异常
            raise ValueError("No valid features to collate after filtering items with errors or missing images.")

        # 从有效特征中提取数据
        images = [f["image"] for f in valid_features]
        instruction_texts = [f["instruction_text"] for f in valid_features]
        target_texts = [f["target_text"] for f in valid_features]

        # 准备 processor 输入
        # Qwen-VL 的 processor 可能需要特定的格式或使用 apply_chat_template
        # 这里我们根据通用做法，将指令和目标合并，并添加 EOS token 到目标末尾
        # 需要确保 processor.tokenizer.eos_token 是有效的
        eos_token = processor.tokenizer.eos_token if processor.tokenizer.eos_token else ""
        
        full_texts_for_processing = [
            instr + tgt + eos_token 
            for instr, tgt in zip(instruction_texts, target_texts)
        ]
        
        # print(f"[Collator PID {os.getpid()}] Sample full text for processing (0): '{full_texts_for_processing[0][:200]}...'")

        try:
            batch = processor(
                text=full_texts_for_processing,
                images=images,
                padding=True,          # Padding to max length in batch
                truncation=True,       # Truncate if exceeds model max length
                return_tensors="pt"
            )
        except Exception as e_proc:
            # print(f"[Collator PID {os.getpid()}] Error during processor call: {e_proc}")
            # print(traceback.format_exc())
            raise

        labels = batch["input_ids"].clone()

        # 核心逻辑：在 labels 中 mask掉 instruction_text 部分，以及 padding 和特殊 image token
        for i in range(len(valid_features)):
            # 1. 确定 instruction_text 在当前样本的 tokenized input_ids 中的长度
            #    注意：processor(...) 的分词行为可能与单独调用 processor.tokenizer(...) 有细微差别，
            #    特别是关于特殊token（如BOS）的添加。
            #    一种策略是只对 instruction_text 分词（不加特殊token），然后看 batch["input_ids"][i] 是否以 BOS 开头。
            
            instruction_only_tokens = processor.tokenizer(instruction_texts[i], add_special_tokens=False).input_ids
            len_instruction_tokens = len(instruction_only_tokens)
            
            # 检查 batch["input_ids"][i] 的第一个 token 是否为 BOS token
            # 并相应地调整掩码的起始长度
            actual_mask_len = len_instruction_tokens
            if processor.tokenizer.bos_token_id is not None and \
               batch["input_ids"][i][0] == processor.tokenizer.bos_token_id:
                # print(f"[Collator PID {os.getpid()}] Detected BOS token at start of input_ids for sample {i}. Adjusting mask.")
                actual_mask_len += 1 # Mask the BOS token as well as it's part of the prompt
            
            # Mask instruction part
            # print(f"[Collator PID {os.getpid()}] Masking {actual_mask_len} tokens for instruction part of sample {i}.")
            labels[i, :actual_mask_len] = -100

        # 2. Mask padding tokens
        # print(f"[Collator PID {os.getpid()}] Masking padding tokens (ID: {processor.tokenizer.pad_token_id}).")
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # 3. Mask special image tokens (来自用户提供的代码片段)
        #    需要确认 Qwen2_5_VLProcessor 是否真的有 image_start_token 和 image_end_token 属性
        #    以及这些 token 是否应该在 labels 中被忽略。
        #    通常，如果图像信息是通过 pixel_values 传入，文本中的特殊图像标记可能用于定位，但不参与loss计算。
        if hasattr(processor, "image_start_token_id") and hasattr(processor, "image_end_token_id"):
            # print(f"[Collator PID {os.getpid()}] Masking image placeholder tokens.")
            # 假设这些属性直接是 token ID
            img_start_id = processor.image_start_token_id
            img_end_id = processor.image_end_token_id
            if img_start_id is not None: labels[labels == img_start_id] = -100
            if img_end_id is not None: labels[labels == img_end_id] = -100
        elif hasattr(processor.tokenizer, "img_start_id") and hasattr(processor.tokenizer, "img_end_id"):
            # 有些模型的 tokenizer 可能直接有这些 ID，例如 qwen tokenizer
            img_start_id = processor.tokenizer.img_start_id
            img_end_id = processor.tokenizer.img_end_id
            if img_start_id is not None: labels[labels == img_start_id] = -100
            if img_end_id is not None: labels[labels == img_end_id] = -100
        elif hasattr(processor.tokenizer, "image_token_index") and processor.tokenizer.image_token_index is not None:
            # 旧版的一些 VLM processor 可能用单个 image_token_index
            # print(f"[Collator PID {os.getpid()}] Masking single image_token_index (ID: {processor.tokenizer.image_token_index}).")
            labels[labels == processor.tokenizer.image_token_index] = -100
        else:
            # print(f"[Collator PID {os.getpid()}] Image placeholder token IDs not found on processor or tokenizer for masking.")
            pass # 如果没有明确的图像token，则不执行此掩码
        
        batch["labels"] = labels
        # print(f"[Collator PID {os.getpid()}] Collation complete. Batch keys: {list(batch.keys())}")
        return batch

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
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        remove_unused_columns=False
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