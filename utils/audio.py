import logging
import requests
from tenacity import retry


@retry
def download_audio(url, save_path, referer: str = "https://www.zdic.net/"):
    try:
        logging.info(f"Downloading audio from {url} to {save_path}")

        headers = {
            "Referer": referer,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        }
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
    
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)

        logging.info(f"Audio downloaded successfully to {save_path}")
    
    except Exception as e:
        logging.error(f"Error downloading audio: {e}")
        raise e
