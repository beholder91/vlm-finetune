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
from accelerate.utils import InitProcessGroupKwargs
import datetime

# 导入数据处理模块
from data_prepare import create_dataloader

# 训练参数 - 可直接修改
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = "./rolmocr_output"
WANDB_PROJECT = "RolmOCR-finetune"
MAX_SAMPLES = 256  # 设置为None表示使用全部样本
EPOCHS = 3
BATCH_SIZE = 2  # 每个GPU的批处理大小
LEARNING_RATE = 3e-5
USE_FP16 = True  # 加回FP16设置
LOGGING_STEPS = 1
SAVE_STEPS = 500
NUM_WORKERS = 0  # 数据加载的线程数
GRADIENT_ACCUMULATION_STEPS = 16  # 加回梯度累积步数设置

def main():
    # 1. 初始化 Accelerator
    timeout_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # 明确设置数值，不使用"auto"
        log_with="wandb",
        kwargs_handlers=[timeout_kwargs]
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
        valid_features = []
        for i, feature in enumerate(features):
            if feature.get("__error__") is not None:
                continue
            if feature.get("image") is None:
                continue
            valid_features.append(feature)

        if not valid_features:
            raise ValueError("No valid features to collate after filtering items with errors or missing images.")

        images = [f["image"] for f in valid_features]
        
        batched_messages = []
        target_texts_for_masking = [] 

        for feature in valid_features:
            instruction_text = feature["instruction_text"]
            target_text = feature["target_text"]
            
            current_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"}, 
                        {"type": "text", "text": instruction_text}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": target_text}]
                }
            ]
            batched_messages.append(current_messages)
            target_texts_for_masking.append(target_text)

        texts_for_processing = [
            processor.apply_chat_template(
                messages_for_sample, 
                tokenize=False, 
                add_generation_prompt=False
            ) 
            for messages_for_sample in batched_messages
        ]
        
        try:
            batch = processor(
                text=texts_for_processing,
                images=images,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e_proc:
            # print(f"[Collator PID {os.getpid()}] Error during processor call: {e_proc}")
            # print(traceback.format_exc())
            raise

        labels = batch["input_ids"].clone()

        # Mask prompt part in labels
        for i in range(len(valid_features)):
            user_instruction_text = valid_features[i]["instruction_text"]
            assistant_target_text = target_texts_for_masking[i]

            prompt_only_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_instruction_text}
                    ]
                }
            ]
            text_of_prompt_to_mask = processor.apply_chat_template(
                prompt_only_messages,
                tokenize=False,
                add_generation_prompt=True 
            )
            
            prompt_tokens_for_masking = processor.tokenizer(text_of_prompt_to_mask, add_special_tokens=False).input_ids
            len_prompt_mask = len(prompt_tokens_for_masking)

            current_mask_len = 0
            if processor.tokenizer.bos_token_id is not None and \
               batch["input_ids"][i][0] == processor.tokenizer.bos_token_id:
                if batch["input_ids"][i][1 : 1 + len_prompt_mask].tolist() == prompt_tokens_for_masking:
                    current_mask_len = 1 + len_prompt_mask
                else:
                    # Fallback: find target tokens by tokenizing target_text separately
                    target_only_tokens = processor.tokenizer(assistant_target_text, add_special_tokens=False).input_ids
                    found_target_start = -1
                    for k_search in range(len(batch["input_ids"][i]) - len(target_only_tokens) + 1):
                        if batch["input_ids"][i][k_search : k_search + len(target_only_tokens)].tolist() == target_only_tokens:
                            if k_search + len(target_only_tokens) < len(batch["input_ids"][i]):
                                next_token = batch["input_ids"][i][k_search + len(target_only_tokens)]
                                if next_token == processor.tokenizer.eos_token_id or next_token == processor.tokenizer.pad_token_id:
                                    found_target_start = k_search
                                    break
                            else: 
                                found_target_start = k_search
                                break
                    if found_target_start != -1:
                        current_mask_len = found_target_start
                    else: 
                        # Fallback to a simpler heuristic if robust target search fails
                        instruction_tokens_for_fallback = processor.tokenizer(user_instruction_text, add_special_tokens=False).input_ids
                        current_mask_len = len(instruction_tokens_for_fallback)
                        if processor.tokenizer.bos_token_id is not None and batch["input_ids"][i][0] == processor.tokenizer.bos_token_id:
                            current_mask_len += 1
            else: 
                if batch["input_ids"][i][:len_prompt_mask].tolist() == prompt_tokens_for_masking:
                    current_mask_len = len_prompt_mask
                else: 
                    # Fallback for no BOS case
                    instruction_tokens_for_fallback = processor.tokenizer(user_instruction_text, add_special_tokens=False).input_ids
                    current_mask_len = len(instruction_tokens_for_fallback)
            labels[i, :current_mask_len] = -100
        
        # Mask padding tokens
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        else:
            # print(f"[Collator PID {os.getpid()}] Warning: processor.tokenizer.pad_token_id is None.")
            pass # Potentially use attention_mask if pad_token_id is None and padding exists

        # Mask special vision tokens (e.g., <|image_pad|>)
        # These should ideally be part of the prompt and masked already, but explicit masking is safer.
        qwen_vision_tokens_to_mask_str = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"]
        for token_str in qwen_vision_tokens_to_mask_str:
            token_id = processor.tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(token_id, int) and token_id != processor.tokenizer.unk_token_id : 
                labels[labels == token_id] = -100
        
        batch["labels"] = labels
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
        gradient_checkpointing=False,
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