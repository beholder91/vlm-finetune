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

# fitz.TOOLS.mupdf_display_errors(False) # <--- 移除

# 配置参数 - 可以直接在此修改
LOCAL_DATASET_PATH = "./olmOCR-mix-0225/train-s2pdf.parquet"
PDF_DIR = "./olmOCR-mix-0225/pdfs"
MAX_SIDE = 1024  # 图像最大边长

ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_image(pdf_id, rotation_prob=0.15, max_side=MAX_SIDE):
    """处理单个图像，使用pdftoppm和PIL，确保所有图像具有相同尺寸"""
    print(f"[PID {os.getpid()}] process_image: 开始处理 pdf_id: {pdf_id} using pdftoppm")
    pdf_path = os.path.join(PDF_DIR, f"{pdf_id}.pdf")
    print(f"[PID {os.getpid()}] process_image: 构造的 PDF 路径: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"[PID {os.getpid()}] process_image: 错误 - PDF 文件不存在: {pdf_path}, pdf_id: {pdf_id}")
        return {"success": False, "error": f"PDF file not found: {pdf_path}", "pdf_id": pdf_id}

    temp_ppm_file = None
    try:
        # 1. 使用pdftoppm将PDF第一页转换为PPM图像到临时文件
        # pdftoppm会自动为输出文件名添加页码后缀，如 "prefix-000001.ppm"
        # 我们需要一个prefix，然后找到实际生成的文件名
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tmp_out:
            temp_ppm_prefix = tmp_out.name.rsplit('.', 1)[0] # 获取不带后缀的路径作为prefix
            # print(f"[PID {os.getpid()}] process_image: 临时PPM文件前缀: {temp_ppm_prefix}")

        # pdftoppm通常输出到stdout或指定文件。使用 -f 1 -l 1 仅处理第一页
        # 为了获取文件名，最好直接指定输出文件名而不是依赖stdout
        # 构建命令: pdftoppm -f 1 -l 1 -r 200 {pdf_path} {temp_ppm_prefix}
        # -r 200 设置DPI为200，与之前fitz的get_pixmap(dpi=200)对应
        cmd = [
            "pdftoppm",
            "-f", "1",     # First page
            "-l", "1",     # Last page
            "-r", "200",   # DPI
            pdf_path,      # Input PDF
            temp_ppm_prefix # Output PPM file prefix
        ]
        
        print(f"[PID {os.getpid()}] process_image: 执行命令: {' '.join(cmd)}")
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
            # 清理可能的临时文件前缀(如果NamedTemporaryFile创建了空文件)
            if os.path.exists(temp_ppm_prefix + ".ppm") and temp_ppm_prefix.endswith(tmp_out.name.rsplit('.', 1)[0]): # 确保是我们创建的
                 try:
                    os.remove(temp_ppm_prefix + ".ppm")
                 except OSError: pass
            return {"success": False, "error": error_message, "pdf_id": pdf_id}

        temp_ppm_file = generated_ppm_path # 用于finally中删除
        # print(f"[PID {os.getpid()}] process_image: PPM 文件已生成: {temp_ppm_file}")

        # 2. 使用PIL读取PPM图像
        img = Image.open(temp_ppm_file).convert("RGB") # 确保是RGB
        
        # print(f"[PID {os.getpid()}] process_image (pdf_id: {pdf_id}): 成功从PPM加载图像，原始尺寸: {img.size}")

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
        
        # print(f"[PID {os.getpid()}] process_image (pdf_id: {pdf_id}): 成功处理图像，最终尺寸: {background.size}")
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
                # print(f"[PID {os.getpid()}] process_image: 临时PPM文件已删除: {temp_ppm_file}")
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
        print(f"[PID {os.getpid()}] DynamicOCRDataset 初始化完成，数据集大小: {len(self.ds)}")
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        print(f"[PID {os.getpid()}] __getitem__: 开始处理索引 {idx}")
        try:
            example = self.ds[idx]
            print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}): 获取到的样本: {example}")

            if not example or not isinstance(example, dict):
                print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}): 错误 - 样本为空或非字典类型: {type(example)}")
                return {
                    "image_bytes": None, # 修改为 image_bytes
                    "image_is_bytes": True,
                    "input_text": "ERROR_INVALID_SAMPLE_STRUCTURE",
                    "text": "ERROR_INVALID_SAMPLE_STRUCTURE",
                    "id": f"ERROR_IDX_{idx}_INVALID_SAMPLE",
                    "width": self.max_side,
                    "height": self.max_side,
                    "__error__": "Invalid sample structure"
                }

            pdf_id_to_use = example.get("id")
            if not pdf_id_to_use:
                print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}): 错误 - 样本中缺少 'id' 键: {example}")
                return {
                    "image_bytes": None, # 修改为 image_bytes
                    "image_is_bytes": True,
                    "input_text": "ERROR_MISSING_ID_IN_SAMPLE",
                    "text": "ERROR_MISSING_ID_IN_SAMPLE",
                    "id": f"ERROR_IDX_{idx}_MISSING_ID",
                    "width": self.max_side,
                    "height": self.max_side,
                    "__error__": "Missing 'id' in sample"
                }
            
            print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}): 使用 pdf_id: {pdf_id_to_use}")
            image_data_result = process_image(pdf_id_to_use, max_side=self.max_side)
            # Log a summary of process_image result, showing type of image
            log_image_data_result = {k: v if k != 'image' else (type(v).__name__ if v is not None else None) for k, v in image_data_result.items()}
            print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_to_use}): process_image 结果: {log_image_data_result}")


            response_str = example.get("response")
            if not response_str or not isinstance(response_str, str):
                print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_to_use}): 警告 - 'response' 字段为空或非字符串: {response_str}")
                natural_text = "ERROR_EMPTY_OR_INVALID_RESPONSE"
            else:
                try:
                    response_json = json.loads(response_str)
                    natural_text = response_json.get("natural_text", "")
                    if not natural_text:
                        natural_text = "PLACEHOLDER_EMPTY_NATURAL_TEXT"
                except json.JSONDecodeError:
                    print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_to_use}): 错误 - 'response' 字段JSON解码失败: {response_str[:100]}...")
                    natural_text = "ERROR_JSON_DECODE_RESPONSE"

            input_text_prompt = "请对图片内容进行详细的OCR识别，包括所有文字和排版信息。"
            
            image_bytes_to_return = None
            image_is_bytes_flag = True # Always true now

            if image_data_result.get("success") and isinstance(image_data_result.get("image"), Image.Image):
                pil_image = image_data_result.get("image")
                try:
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    image_bytes_to_return = buffer.getvalue()
                    print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_to_use}): PIL Image 转换为 PNG bytes, 大小: {len(image_bytes_to_return)}")
                except Exception as e_save_bytes:
                    print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_to_use}): 错误 - PIL Image 转换为 PNG bytes 失败: {e_save_bytes}")
                    print(traceback.format_exc())
                    # image_bytes_to_return remains None
            
            final_sample = {
                "image_bytes": image_bytes_to_return,
                "image_is_bytes": image_is_bytes_flag,
                "input_text": input_text_prompt,
                "text": natural_text,
                "id": str(pdf_id_to_use),
                "width": image_data_result.get("width", self.max_side) if image_data_result.get("success") else self.max_side,
                "height": image_data_result.get("height", self.max_side) if image_data_result.get("success") else self.max_side,
            }

            if not image_data_result.get("success") or image_bytes_to_return is None:
                # Ensure __error__ is set if image processing failed or byte conversion failed
                error_detail = image_data_result.get("error", "Unknown image processing error")
                if image_data_result.get("success") and image_bytes_to_return is None: # Conversion failed
                     error_detail = "PIL Image to PNG bytes conversion failed"
                final_sample["__error__"] = error_detail
                print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}, pdf_id: {pdf_id_to_use}): 图像处理或转换字节流失败，错误: {final_sample['__error__']}")
            
            return dict(final_sample)

        except Exception as e_outer:
            print(f"[PID {os.getpid()}] __getitem__ (idx: {idx}): 发生严重外部错误: {e_outer}")
            print(traceback.format_exc())
            return {
                "image_bytes": None, # 修改为 image_bytes
                "image_is_bytes": True,
                "input_text": "ERROR_OUTER_EXCEPTION_IN_GETITEM",
                "text": "ERROR_OUTER_EXCEPTION_IN_GETITEM",
                "id": f"ERROR_IDX_{idx}_OUTER_EXCEPTION",
                "width": self.max_side,
                "height": self.max_side,
                "__error__": f"Outer exception: {e_outer}"
            }

def create_dataloader(batch_size, num_workers, shuffle=True, max_samples=None, data_path=LOCAL_DATASET_PATH, pdf_dir=PDF_DIR, max_side=MAX_SIDE):
    print(f"加载数据集: {data_path}，共{max_samples if max_samples is not None else '全部'}个样本")
    dataset = load_dataset("parquet", data_files=data_path, split=f"train[:{max_samples}]" if max_samples else "train")
    
    processed_dataset = DynamicOCRDataset(dataset, pdf_dir=pdf_dir, max_side=max_side)
    
    return DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory=True, # pin_memory=True 通常与 num_workers > 0 结合使用效果更佳
        drop_last=True
    ), processed_dataset

def render_pdf_to_base64png(pdf_path: str, target_longest_dim: int = 2048) -> str:
    """使用 pdftoppm 和 PIL 将 PDF 首页渲染为指定最大边长的 Base64 PNG 字符串。"""
    print(f"[PID {os.getpid()}] render_pdf_to_base64png: 开始处理 PDF: {pdf_path} to base64, target_longest_dim: {target_longest_dim}")
    
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
        
        print(f"[PID {os.getpid()}] render_pdf_to_base64png: 执行命令: {' '.join(cmd_render)}")
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
        # print(f"[PID {os.getpid()}] render_pdf_to_base64png: PPM 文件已生成: {temp_ppm_file_render}")

        # 2. 使用PIL读取PPM图像
        img = Image.open(temp_ppm_file_render).convert("RGB")

        # 3. 缩放图像以适应 target_longest_dim，保持宽高比
        #    原fitz逻辑是：scale = target_longest_dim / max(width, height)，然后get_pixmap(matrix=fitz.Matrix(scale, scale))
        #    这等同于将图像缩放到最长边为 target_longest_dim，然后粘贴到黑色背景
        
        # 使用Pillow的thumbnail进行等比例缩放，使其最长边不超过target_longest_dim
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
        
        # 主动调用垃圾回收 (可选，但保留原逻辑)
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

if __name__ == "__main__":
    # 测试 num_workers=0 的情况
    print("\n--- Testing with num_workers=0 ---")
    # 使用较小的 max_samples 以便快速测试
    dataloader_nw0, dataset_nw0 = create_dataloader(
        batch_size=2, 
        num_workers=0, 
        max_samples=4,  # Reduced for faster testing
        data_path=LOCAL_DATASET_PATH, 
        pdf_dir=PDF_DIR, 
        max_side=MAX_SIDE
    )
    print(f"创建了动态数据加载器 (nw=0)，数据集大小: {len(dataset_nw0)}")
    print("加载一个批次数据 (nw=0)...")
    
    # nw=0 的详细检查
    processed_batches_nw0 = 0
    for i, batch_data_nw0 in enumerate(dataloader_nw0):
        print(f"\n--- Batch {i} (nw=0) ---")
        print(f"Type of batch_data: {type(batch_data_nw0)}")
        if isinstance(batch_data_nw0, dict):
            print(f"Keys in batch_data: {list(batch_data_nw0.keys())}")
            
            # 详细检查 image_bytes
            if "image_bytes" in batch_data_nw0:
                print(f"Length of image_bytes list: {len(batch_data_nw0['image_bytes'])}")
                if batch_data_nw0['image_bytes'] and batch_data_nw0['image_bytes'][0] is not None:
                    first_image_bytes_sample = batch_data_nw0['image_bytes'][0]
                    print(f"Type of first image_bytes sample: {type(first_image_bytes_sample)}")
                    if isinstance(first_image_bytes_sample, bytes):
                        print(f"Size of first image_bytes sample: {len(first_image_bytes_sample)} bytes")
                        try:
                            img = Image.open(io.BytesIO(first_image_bytes_sample))
                            print(f"  Successfully decoded first image_bytes to PIL Image. Size: {img.size}, Mode: {img.mode}")
                        except Exception as e_decode:
                            print(f"  ERROR decoding first image_bytes: {e_decode}")
                    else:
                        print(f"  First image_bytes sample is NOT bytes, but {type(first_image_bytes_sample)}")
                else:
                    print("  First image_bytes sample is None or list is empty.")
            else:
                print("  'image_bytes' key NOT found in batch_data.")

            # 检查其他所有预期的键
            expected_keys = ["text", "id", "input_text", "image_is_bytes", "width", "height", "__error__"]
            for k in expected_keys:
                if k in batch_data_nw0:
                    # Safely access the first element if it's a list and not empty
                    value_list = batch_data_nw0[k]
                    example_value = "N/A"
                    if isinstance(value_list, list):
                        if value_list: # List is not empty
                            example_value = str(value_list[0])[:100] # Take first item, convert to str, truncate
                        else: # List is empty
                            example_value = "[] (empty list)"
                    else: # Not a list
                         example_value = str(value_list)[:100]


                    print(f"  Key '{k}': Present. Type: {type(value_list)}. Example/Value[0]: {example_value}")
                elif k != "__error__": # __error__ is optional
                    print(f"  Key '{k}': MISSING")
                elif k == "__error__" and k not in batch_data_nw0:
                     print(f"  Key '{k}': Not present (which is good if no error)")


        else:
            print(f"batch_data (nw=0) is not a dict, it's: {batch_data_nw0}")
        
        processed_batches_nw0 += 1
        if processed_batches_nw0 >= 1: # 只检查第一个批次进行详细打印
            break
    if processed_batches_nw0 == 0:
        print("  (nw=0) No batches were processed from dataloader_nw0.")

    # 测试 num_workers=2 (或您脚本中默认的) 的情况
    print("\n--- Testing with num_workers=2 ---")
    dataloader_nw2, dataset_nw2 = create_dataloader(
        batch_size=2, 
        num_workers=2, 
        max_samples=4, # Reduced for faster testing
        data_path=LOCAL_DATASET_PATH, 
        pdf_dir=PDF_DIR, 
        max_side=MAX_SIDE
    )
    print(f"创建了动态数据加载器 (nw=2)，数据集大小: {len(dataset_nw2)}")
    print("加载一个批次数据 (nw=2)...")
    
    processed_batches_nw2 = 0
    for i, batch_data_nw2 in enumerate(dataloader_nw2):
        print(f"\n--- Batch {i} (nw=2) ---")
        print(f"Type of batch_data: {type(batch_data_nw2)}")
        if isinstance(batch_data_nw2, dict):
            print(f"Keys in batch_data: {list(batch_data_nw2.keys())}")
            
            if "image_bytes" in batch_data_nw2:
                print(f"Length of image_bytes list: {len(batch_data_nw2['image_bytes'])}")
                if batch_data_nw2['image_bytes'] and batch_data_nw2['image_bytes'][0] is not None:
                    first_image_bytes_sample_nw2 = batch_data_nw2['image_bytes'][0]
                    print(f"Type of first image_bytes sample: {type(first_image_bytes_sample_nw2)}")
                    if isinstance(first_image_bytes_sample_nw2, bytes):
                        print(f"Size of first image_bytes sample: {len(first_image_bytes_sample_nw2)} bytes")
                        try:
                            img_nw2 = Image.open(io.BytesIO(first_image_bytes_sample_nw2))
                            print(f"  Successfully decoded first image_bytes to PIL Image. Size: {img_nw2.size}, Mode: {img_nw2.mode}")
                        except Exception as e_decode_nw2:
                            print(f"  ERROR decoding first image_bytes (nw=2): {e_decode_nw2}")
                    else:
                        print(f"  First image_bytes sample (nw=2) is NOT bytes, but {type(first_image_bytes_sample_nw2)}")

                else:
                    print("  First image_bytes sample (nw=2) is None or list is empty.")
            else:
                 print("  'image_bytes' key NOT found in batch_data (nw=2).")
            
            # 简化的其他键检查 (nw=2)
            for k_nw2 in ["text", "id", "input_text"]:
                 if k_nw2 in batch_data_nw2:
                     print(f"  Key '{k_nw2}' (nw=2): Present")
                 else:
                     print(f"  Key '{k_nw2}' (nw=2): MISSING")

        else:
            print(f"batch_data (nw=2) is not a dict, it's: {batch_data_nw2}")

        processed_batches_nw2 +=1
        if processed_batches_nw2 >= 1: # 只检查第一个批次
            break
    if processed_batches_nw2 == 0:
        print("  (nw=2) No batches were processed from dataloader_nw2.") 