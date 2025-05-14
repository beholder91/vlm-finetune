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
        print(f"[Collator PID {os.getpid()}] Inspecting {len(features)} features in this batch.")
        valid_features = []
        problematic_feature_details = []

        for i, feature in enumerate(features):
            feature_id_str = "UnknownID" # Default ID string
            is_dict = isinstance(feature, dict)
            
            if is_dict:
                feature_id_str = str(feature.get('id', f'UnknownID_InDict_Idx_{i}'))

                # 新增：处理图像字节流
                if feature.get("image_is_bytes") and feature.get("image_bytes") is not None:
                    # print(f"[Collator PID {os.getpid()}] Feature {i} (id: {feature_id_str}) contains image_bytes. Attempting to decode.")
                    try:
                        img_bytes = feature["image_bytes"]
                        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        feature["image"] = pil_image # 将解码后的图像存入 "image" 键
                        del feature["image_bytes"]
                        del feature["image_is_bytes"]
                        # print(f"[Collator PID {os.getpid()}] Feature {i} (id: {feature_id_str}) successfully decoded image_bytes to PIL.Image.")
                    except Exception as e_decode:
                        problematic_feature_details.append(f"Feature {i} (id: {feature_id_str}) failed to decode image_bytes: {e_decode}. Skipping.")
                        # print(traceback.format_exc()) # 可选：打印完整堆栈以调试解码错误
                        continue # 跳过这个损坏的特征
                elif feature.get("image_is_bytes") and feature.get("image_bytes") is None:
                    # __getitem__ 表明是字节流，但内容为 None (可能是在 __getitem__ 中处理失败)
                    problematic_feature_details.append(f"Feature {i} (id: {feature_id_str}) marked as image_bytes but bytes are None. Error from __getitem__: {feature.get('__error__', 'Unknown error in __getitem__')}. Skipping.")
                    continue

            if not is_dict:
                problematic_feature_details.append(f"Feature {i} (id: {feature_id_str}) is not a dict, type: {type(feature)}. Skipping.")
                continue

            # 现在我们知道 feature 是一个 dict
            # 检查是否包含必须的键，包括 "image" (此时应该已被image_bytes转换而来)
            keys = feature.keys()
            missing_keys = []
            if "image" not in keys: missing_keys.append("image") # 必须检查 "image"，而不是 "image_bytes"
            if "input_text" not in keys: missing_keys.append("input_text")
            if "text" not in keys: missing_keys.append("text")

            if missing_keys:
                # 如果 image_is_bytes 为 True 但 image_bytes 为 None，之前已处理并跳过
                # 此处 missing_keys 包含 "image" 意味着原始样本就没有 image_is_bytes 标记，或者解码失败后 continue 了
                problematic_feature_details.append(f"Feature {i} (id: {feature_id_str}) is missing keys: {missing_keys}. All keys: {list(keys)}. Error from __getitem__: {feature.get('__error__', 'N/A')}. Skipping.")
                continue
            
            if feature["image"] is None: # 此时的 feature["image"] 应该是 PIL Image 或 None
                problematic_feature_details.append(f"Feature {i} (id: {feature_id_str}) has 'image' key, but its value is None. Error from __getitem__: {feature.get('__error__', 'Image became None')}. Skipping.")
                continue
            
            # If we reach here, the feature is considered valid for basic structure
            valid_features.append(feature)

        if problematic_feature_details:
            print(f"[Collator PID {os.getpid()}] Problems found in batch features:")
            for detail in problematic_feature_details:
                print(f"  - {detail}")
        
        if not valid_features:
            print(f"[Collator PID {os.getpid()}] CRITICAL WARNING: No valid features left in batch after filtering. Original batch size: {len(features)}. Returning empty dict, expect a crash in Trainer.")
            # To avoid immediate crash *in collator* with empty lists for processor:
            # Option 1: Raise an error, which will be caught by the main try-except in train.py
            raise ValueError(f"Custom_data_collator: No valid features in batch after filtering. Original size: {len(features)}.")
            # Option 2: Return {} - but trainer.train() will likely fail more obscurely.
            # return {} 

        # Proceed with valid features
        try:
            images = [f["image"] for f in valid_features]
            input_texts = [f["input_text"] for f in valid_features]
            target_texts = [f["text"] for f in valid_features]
            
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
        except Exception as e_proc:
            print(f"[Collator PID {os.getpid()}] Error during processor application in collator (with {len(valid_features)} valid features): {e_proc}")
            print(traceback.format_exc())
            # Propagate error. This will be caught by the main try-except in train.py's main function.
            raise e_proc

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
        dataloader_num_workers=0
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