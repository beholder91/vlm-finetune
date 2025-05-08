import requests
import base64
import json

def send_image_to_vllm(image_path):
    """发送本地图片到vLLM服务器并获取描述"""
    # 读取并编码图片
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    # 请求数据
    data = {
        "model": "reducto/RolmOCR",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return the plain text representation of this document as if you were reading it naturally"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }
        ]
    }
    
    # 发送请求
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    # 返回结果
    return response.json()["choices"][0]["message"]["content"]

# 使用示例
if __name__ == "__main__":
    result = send_image_to_vllm("test_data/unusual2.png")
    print(result)
