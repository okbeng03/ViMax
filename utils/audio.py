import logging
import requests
from tenacity import retry


@retry
def download_audio(url, save_path):
    try:
        logging.info(f"Downloading audio from {url} to {save_path}")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
    
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)

        logging.info(f"Audio downloaded successfully to {save_path}")
    
    except Exception as e:
        logging.error(f"Error downloading audio: {e}")
        raise e
