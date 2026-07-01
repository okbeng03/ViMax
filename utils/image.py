import logging
import requests
import base64
import mimetypes
import os
import uuid
from tenacity import retry
from pathlib import Path
from io import BytesIO


@retry(3)
def download_image(url, save_path):
    try:
        logging.info(f"Downloading image from {url} to {save_path}")

        response = requests.get(url, stream=True)
        response.raise_for_status() # Check for HTTP errors

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        logging.info(f"Image downloaded successfully to {save_path}")

    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        raise e


def image_path_to_b64(image_path, mime: bool = True) -> str:
    with open(image_path, 'rb') as image_file:
        b64 = base64.b64encode(image_file.read()).decode('utf-8')

    if mime:
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        return f"data:{mime_type};base64,{b64}"

    return b64


def pil_to_b64(image, mime: bool = True) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    if mime:
        return f"data:image/png;base64,{b64}"

    return b64


def save_base64_image(b64_string, save_path):
    # If the base64 string has a data URL prefix, remove it
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]

    with open(save_path, 'wb') as image_file:
        image_file.write(base64.b64decode(b64_string))

# 百炼图片上传
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

# 凭证缓存（复用凭证避免限流，凭证有效期约5分钟）
_policy_cache: dict = {}

def get_cached_policy(model_name):
    """获取缓存的凭证，凭证有效期约5分钟"""
    import time
    
    cache_key = model_name
    if cache_key in _policy_cache:
        cached = _policy_cache[cache_key]
        # 4分钟内复用凭证，预留1分钟安全时间
        if time.time() - cached['timestamp'] < 240:
            # logging.info(f"复用缓存的上传凭证 for {model_name}")
            return cached['policy_data']
    
    # 获取新凭证
    policy_data = get_upload_policy(model_name)
    _policy_cache[cache_key] = {
        'policy_data': policy_data,
        'timestamp': time.time()
    }
    return policy_data

def get_upload_policy(model_name):
    """获取文件上传凭证"""

    url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {
        "Authorization": f"Bearer {dashscope_api_key}",
        "Content-Type": "application/json"
    }
    params = {
        "action": "getPolicy",
        "model": model_name
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to get upload policy: {response.text}")
    
    return response.json()['data']

def upload_file_to_oss(policy_data, file_path):
    """将文件上传到临时存储OSS"""

    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}_{uuid.uuid4()}"
    
    with open(file_path, 'rb') as file:
        files = {
            'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
            'Signature': (None, policy_data['signature']),
            'policy': (None, policy_data['policy']),
            'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
            'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
            'key': (None, key),
            'success_action_status': (None, '200'),
            'file': (file_name, file)
        }
        
        response = requests.post(policy_data['upload_host'], files=files)
        if response.status_code != 200:
            raise Exception(f"Failed to upload file: {response.text}")
    
    return f"oss://{key}"

def upload_file_and_get_url(model_name, file_path):
    """上传文件并获取URL（自动复用凭证）"""

    # 1. 获取缓存的凭证（自动复用，凭证有效期约5分钟）
    policy_data = get_cached_policy(model_name)
    # 2. 上传文件到OSS
    oss_url = upload_file_to_oss(policy_data, file_path)
    
    return oss_url
