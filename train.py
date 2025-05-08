#!/usr/bin/env python3
# train.py — RolmOCR 全量微调脚本，集成 HF Trainer + W&B + TensorBoard

import os
import random
import io
import requests
from datasets import load_dataset
from PIL import Image
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers.integrations import WandbCallback, TensorBoardCallback
import wandb
import fitz  # PyMuPDF

def preprocess_example(example, rotation_prob=0.15, max_side=1024):
    # 从URL获取PDF并提取特定页面
    response = requests.get(example["url"])
    pdf_data = io.BytesIO(response.content)
    pdf_doc = fitz.open(stream=pdf_data, filetype="pdf")
    page = pdf_doc[example["page_number"] - 1]  # 页码从0开始索引
    pix = page.get_pixmap()
    img_data = pix.tobytes("ppm")
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    # 15% 概率随机旋转输入图像
    if random.random() < rotation_prob:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, expand=True)
    # Resize: 最长边为 max_side
    w, h = img.size
    scale = max_side / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)))
    example["img_tensor"] = torch.tensor(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
         .view(h, w, 3)
         .permute(2, 0, 1)),
        dtype=torch.float32
    ) / 255.0
    example["labels"] = example["response"]["raw_text"]
    return example

def main():
    # 1. 环境变量与 W&B 初始化
    os.environ["WANDB_PROJECT"] = "RolmOCR-finetune"
    wandb.init()

    # 2. 参数与超参
    model_id    = "Qwen/Qwen2.5-VL-2B-Instruct"
    dataset_id  = "allenai/olmOCR-mix-0225"
    output_dir  = "./rolmocr_output"
    epochs      = 3
    batch_size  = 1
    lr          = 3e-5
    max_len     = 4096

    # 3. 加载数据集并预处理
    ds = load_dataset(dataset_id, "00_documents", split="train_s2pdf")  # Parquet 格式 OCR 数据集:contentReference[oaicite:6]{index=6}
    ds = ds.map(lambda x: preprocess_example(x), remove_columns=ds.column_names)

    # 4. 分词器与模型加载
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)  # 远程自定义加载:contentReference[oaicite:7]{index=7}
    model     = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    # 5. 数据 collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # 6. TrainingArguments 配置（混合精度 + W&B + TensorBoard）
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        fp16=True,  # 混合精度训练:contentReference[oaicite:8]{index=8}
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        report_to=["wandb","tensorboard"],  # 一行开启 W&B + TB:contentReference[oaicite:9]{index=9}
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="loss",
    )

    # 7. Trainer 实例化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[WandbCallback, TensorBoardCallback]
    )

    # 8. 开始训练
    trainer.train()

    # 9. 保存模型与 Tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"训练完成，模型保存在 {output_dir}")

if __name__ == "__main__":
    main()
