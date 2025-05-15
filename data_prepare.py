import os
import io
import json
import random
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import gc
import base64
import traceback
import subprocess
import tempfile
import torch

# 配置参数 - 可以直接在此修改
# LOCAL_DATASET_PATH = "./olmOCR-mix-0225/train-s2pdf.parquet"
LOCAL_DATASET_PATH = "./olmOCR-mix-0225/cleaned_train_data.parquet"
PDF_DIR = "./olmOCR-mix-0225/pdfs"
MAX_SIDE = 448  # 图像最大边长

ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_image(pdf_id, rotation_prob=0.15, max_side=MAX_SIDE):
    """处理单个图像，使用pdftoppm和PIL，确保所有图像具有相同尺寸"""
    pdf_path = os.path.join(PDF_DIR, f"{pdf_id}.pdf")

    if not os.path.exists(pdf_path):
        print(f"[PID {os.getpid()}] process_image: 错误 - PDF 文件不存在: {pdf_path}, pdf_id: {pdf_id}")
        return {"success": False, "error": f"PDF file not found: {pdf_path}", "pdf_id": pdf_id}

    temp_ppm_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tmp_out:
            temp_ppm_prefix = tmp_out.name.rsplit('.', 1)[0] # 获取不带后缀的路径作为prefix

        cmd = [
            "pdftoppm",
            "-f", "1",     # First page
            "-l", "1",     # Last page
            "-r", "200",   # DPI
            pdf_path,      # Input PDF
            temp_ppm_prefix # Output PPM file prefix
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        generated_ppm_path = f"{temp_ppm_prefix}-000001.ppm" # pdftoppm 0.86+ 的命名格式
        # 兼容旧版 pdftoppm 可能的命名格式如 prefix-1.ppm
        if not os.path.exists(generated_ppm_path):
            generated_ppm_path_alt = f"{temp_ppm_prefix}-1.ppm"
            if os.path.exists(generated_ppm_path_alt):
                generated_ppm_path = generated_ppm_path_alt
            else: # 再尝试一个更简单的后缀
                generated_ppm_path_single_page = f"{temp_ppm_prefix}.ppm" # 如果只有一页且不指定-singlefile，有些版本可能直接用prefix.ppm
                if os.path.exists(generated_ppm_path_single_page):
                     generated_ppm_path = generated_ppm_path_single_page
                else: # 如果命令失败，result.stdout/stderr会有信息
                    pass # generated_ppm_path 将不存在，后续会捕获

        if result.returncode != 0 or not os.path.exists(generated_ppm_path):
            error_message = f"pdftoppm 执行失败. Code: {result.returncode}. Error: {result.stderr.strip()}. stdout: {result.stdout.strip()}. PDF: {pdf_path}"
            if not os.path.exists(generated_ppm_path) and result.returncode == 0 :
                 error_message += f" Output PPM file {generated_ppm_path} not found despite pdftoppm success."
            print(f"[PID {os.getpid()}] process_image: {error_message}")
            if os.path.exists(temp_ppm_prefix + ".ppm") and temp_ppm_prefix.endswith(tmp_out.name.rsplit('.', 1)[0]): # 确保是我们创建的
                 try:
                    os.remove(temp_ppm_prefix + ".ppm")
                 except OSError: pass
            return {"success": False, "error": error_message, "pdf_id": pdf_id}

        temp_ppm_file = generated_ppm_path # 用于finally中删除

        # 2. 使用PIL读取PPM图像
        img = Image.open(temp_ppm_file).convert("RGB") # 确保是RGB
        
        # 3. 后续处理（旋转、缩放）与之前一致
        # 随机旋转（轻微）
        if random.random() < rotation_prob:
            angle = random.uniform(-2, 2) # 随机旋转-2到2度
            img = img.rotate(angle, expand=True, fillcolor='white')

        # 调整图像大小，保持宽高比，使最大边为 max_side
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        # 创建一个指定大小的白色背景
        background = Image.new('RGB', (max_side, max_side), 'white')
        paste_x = (max_side - img.width) // 2
        paste_y = (max_side - img.height) // 2
        background.paste(img, (paste_x, paste_y))
        
        original_width, original_height = img.width, img.height # 注意这里是thumbnail后的尺寸
        
        return {"success": True, "image": background, "width": original_width, "height": original_height, "pdf_id": pdf_id}

    except FileNotFoundError as e_fnf: # 特别处理pdftoppm未找到的情况
        error_msg = f"pdftoppm 命令未找到. 请确保poppler-utils已安装并路径正确. Error: {e_fnf}"
        print(f"[PID {os.getpid()}] process_image: {error_msg}, pdf_id: {pdf_id}")
        print(traceback.format_exc())
        return {"success": False, "error": error_msg, "pdf_id": pdf_id}
    except Exception as e:
        print(f"[PID {os.getpid()}] process_image: 处理 PDF (pdftoppm) 时发生未知错误: {pdf_path}, pdf_id: {pdf_id}, 错误: {e}")
        print(traceback.format_exc())
        return {"success": False, "error": f"Unknown error (pdftoppm): {e}", "pdf_id": pdf_id}
    finally:
        # 清理临时的PPM文件
        if temp_ppm_file and os.path.exists(temp_ppm_file):
            try:
                os.remove(temp_ppm_file)
            except OSError as e_ose:
                print(f"[PID {os.getpid()}] process_image: 删除临时PPM文件失败: {temp_ppm_file}, Error: {e_ose}")
        # 尝试清理原始的空占位文件（如果delete=False导致它残留）
        if os.path.exists(temp_ppm_prefix + ".ppm") and temp_ppm_prefix.endswith(tmp_out.name.rsplit('.', 1)[0]):
            try:
                os.remove(temp_ppm_prefix + ".ppm")
            except OSError: pass

class DynamicOCRDataset(Dataset):
    """动态OCR数据集，实时处理图像数据"""
    
    def __init__(self, hf_dataset, pdf_dir, max_side, processor=None):
        """初始化数据集
        
        Args:
            hf_dataset: HuggingFace dataset
            pdf_dir: PDF文件目录
            max_side: 图像最大边长
        """
        self.ds = hf_dataset
        self.pdf_dir = pdf_dir
        self.max_side = max_side
        # self.processor = processor # 如果在此处使用processor，需要确保它是线程安全的或每个worker一个实例
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        try:
            example = self.ds[idx]
            pdf_id_to_use = example.get("id")

            if not pdf_id_to_use:
                print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}): 跳过 - 样本中缺少 'id' 键.")
                return None

            # 1. 处理图像
            image_data_result = process_image(pdf_id_to_use, max_side=self.max_side)
            
            if not image_data_result.get("success") or not isinstance(image_data_result.get("image"), Image.Image):
                error_detail = image_data_result.get("error", "未知图像处理错误")
                print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_to_use}): 跳过 - 图像处理失败或返回无效图像. 错误: {error_detail}")
                return None
            
            pil_image_for_sample = image_data_result["image"]

            # 2. 处理 response JSON
            response_str = example.get("response")
            natural_text = "" # 初始化为空字符串

            if not response_str or not isinstance(response_str, str):
                print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_to_use}): 跳过 - 'response' 字段为空或非字符串.")
                return None
            
            try:
                response_json = json.loads(response_str)
                natural_text = response_json.get("natural_text", "") # 如果没有natural_text，默认为空字符串
                if not natural_text: # 明确处理 natural_text 为空字符串的情况
                    natural_text = "PLACEHOLDER_EMPTY_NATURAL_TEXT" # 或保持为空，取决于后续处理
            except json.JSONDecodeError as e_json:
                print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_to_use}): 跳过 - 'response' 字段JSON解码失败. Error: {e_json}")
                return None # 跳过此样本
            
            # 如果执行到这里，说明图像和文本都已成功处理
            input_text_prompt = "请对图片内容进行详细的OCR识别，包括所有文字和排版信息。"
            
            return {
                "image": pil_image_for_sample,
                "instruction_text": input_text_prompt,
                "target_text": natural_text,
                "id": str(pdf_id_to_use),
            }

        except Exception as e_outer:
            # 捕获任何在 __getitem__ 内部发生的其他未预料到的错误
            pdf_id_info = example.get("id", "未知ID") if 'example' in locals() else "未知ID"
            print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_info}): 跳过 - 发生意外的外部错误: {e_outer}")
            # traceback.print_exc() # 如果需要详细堆栈跟踪，可以取消注释，但为了简洁，默认注释掉
            return None

def custom_collate_fn(batch):
    valid_samples = []
    for item in batch:
        if item is None: # 如果 __getitem__ 返回了 None
            # print(f"[PID {os.getpid()}] collate_fn: Received a None item, skipping.") # 可选的调试信息
            continue # 跳过这个 None 值
        valid_samples.append(item)

    if not valid_samples:
        # print(f"[PID {os.getpid()}] collate_fn: Entire batch is invalid, returning None.") # 可选的调试信息
        return None # 如果整个批次都是无效的，返回 None

    keys = valid_samples[0].keys()
    collated_batch = {}
    for key in keys:
        collated_batch[key] = [d.get(key) for d in valid_samples]

    return collated_batch

def create_dataloader(batch_size, num_workers, shuffle=True, max_samples=None, data_path=LOCAL_DATASET_PATH, pdf_dir=PDF_DIR, max_side=MAX_SIDE):
    dataset = load_dataset("parquet", data_files=data_path, split=f"train[:{max_samples}]" if max_samples else "train")
    
    processed_dataset = DynamicOCRDataset(dataset, pdf_dir=pdf_dir, max_side=max_side)
    
    return DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        # pin_memory=True, # pin_memory=True 通常与 num_workers > 0 结合使用效果更佳
        drop_last=True
    ), processed_dataset

def render_pdf_to_base64png(pdf_path: str, target_longest_dim: int = 2048) -> str:
    """使用 pdftoppm 和 PIL 将 PDF 首页渲染为指定最大边长的 Base64 PNG 字符串。"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    temp_ppm_file_render = None
    try:
        # 1. 使用pdftoppm将PDF第一页转换为PPM图像到临时文件
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tmp_out_render:
            temp_ppm_prefix_render = tmp_out_render.name.rsplit('.', 1)[0]
        cmd_render = [
            "pdftoppm",
            "-f", "1",
            "-l", "1",
            "-r", "300", # 使用固定高DPI渲染
            pdf_path,
            temp_ppm_prefix_render
        ]
        
        result_render = subprocess.run(cmd_render, capture_output=True, text=True, check=False)

        generated_ppm_path_render = f"{temp_ppm_prefix_render}-000001.ppm"
        if not os.path.exists(generated_ppm_path_render):
            generated_ppm_path_render_alt = f"{temp_ppm_prefix_render}-1.ppm"
            if os.path.exists(generated_ppm_path_render_alt):
                generated_ppm_path_render = generated_ppm_path_render_alt
            else:
                generated_ppm_path_render_single = f"{temp_ppm_prefix_render}.ppm"
                if os.path.exists(generated_ppm_path_render_single):
                    generated_ppm_path_render = generated_ppm_path_render_single
                else:
                    pass

        if result_render.returncode != 0 or not os.path.exists(generated_ppm_path_render):
            error_message_render = f"pdftoppm (render) 执行失败. Code: {result_render.returncode}. Error: {result_render.stderr.strip()}. stdout: {result_render.stdout.strip()}. PDF: {pdf_path}"
            if not os.path.exists(generated_ppm_path_render) and result_render.returncode == 0:
                 error_message_render += f" Output PPM file {generated_ppm_path_render} not found despite pdftoppm success."
            print(f"[PID {os.getpid()}] render_pdf_to_base64png: {error_message_render}")
            if os.path.exists(temp_ppm_prefix_render + ".ppm") and temp_ppm_prefix_render.endswith(tmp_out_render.name.rsplit('.',1)[0]):
                 try: os.remove(temp_ppm_prefix_render + ".ppm")
                 except OSError: pass
            raise RuntimeError(error_message_render)

        temp_ppm_file_render = generated_ppm_path_render

        # 2. 使用PIL读取PPM图像
        img = Image.open(temp_ppm_file_render).convert("RGB")
        img.thumbnail((target_longest_dim, target_longest_dim), Image.Resampling.LANCZOS)
        
        # 创建一个固定大小的黑色背景图像 (原代码是黑色 (0,0,0))
        padded_img = Image.new("RGB", (target_longest_dim, target_longest_dim), (0, 0, 0))
        
        # 将调整大小后的图像粘贴到中心位置
        paste_x = (target_longest_dim - img.width) // 2
        paste_y = (target_longest_dim - img.height) // 2
        padded_img.paste(img, (paste_x, paste_y))
        
        # 保存为PNG并转换为Base64
        buffer = io.BytesIO()
        padded_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        del img
        del padded_img
        gc.collect()
        
        return img_base64
        
    except FileNotFoundError as e_fnf_render:
        error_msg_render = f"pdftoppm 命令未找到 (render_pdf_to_base64png). 请确保poppler-utils已安装并路径正确. Error: {e_fnf_render}"
        print(f"[PID {os.getpid()}] render_pdf_to_base64png: {error_msg_render}")
        raise RuntimeError(error_msg_render) from e_fnf_render
    except Exception as e_render:
        raise RuntimeError(f"PDF转换为Base64 PNG (pdftoppm) 时出错: {pdf_path}, Error: {e_render}") from e_render
    finally:
        if temp_ppm_file_render and os.path.exists(temp_ppm_file_render):
            try:
                os.remove(temp_ppm_file_render)
            except OSError as e_ose_render:
                print(f"[PID {os.getpid()}] render_pdf_to_base64png: 删除临时PPM文件失败: {temp_ppm_file_render}, Error: {e_ose_render}")
        if os.path.exists(temp_ppm_prefix_render + ".ppm") and temp_ppm_prefix_render.endswith(tmp_out_render.name.rsplit('.',1)[0]):
            try: os.remove(temp_ppm_prefix_render + ".ppm")
            except OSError: pass