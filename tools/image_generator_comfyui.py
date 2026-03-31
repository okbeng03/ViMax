# https://yunwu.apifox.cn/api-347960869

import logging
import aiohttp
import json
from websockets.sync.client import connect
from typing import List, Optional
from tenacity import retry, stop_after_attempt
from utils.retry import after_func
from utils.image import image_path_to_b64
from interfaces.image_output import ImageOutput


class ImageGeneratorDoubaoSeedreamYunwuAPI:
    def __init__(
        self,
        api_key: str,
        model: str = "doubao-seedream-4-5-251128",
        base_url: str = "http://127.0.0.1:8188",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model


    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        size: Optional[str] = None,
        **kwargs,
    ) -> ImageOutput:
        """
            size: [1024x1024, 4096x4096]
        """

        logging.info(f"Calling {self.model} to generate image...")

        image = [
            image_path_to_b64(path, mime=True) for path in reference_image_paths
        ]
        
        # TODO:: 调整 workflow
        with open("workflow_api.json") as f:
            workflow = json.load(f)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "sequential_image_generation": "disabled",  # "auto" or "disabled"
            # "sequential_image_generation_options": {
            #     "max_images": 1
            # },
            "response_format": "url",
            "size": size if size is not None else "2048x2048",
            "watermark": False,
        }
        if len(image) > 0:
            payload["image"] = image
            
        headers = {
            # "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/prompt", json=payload, headers=headers) as response:
                    response_json = await response.json()
                    prompt_id = response_json["prompt_id"]
            
                    with connect(self.base_url.replace("http://", "ws://")) as websocket:
                        # websocket.send("")
                        message = websocket.recv()
                        print(f"Received: {message}")
                        
                        # TODO:: 只过滤当前 prompt_id
        except Exception as e:
            logging.error(f"Error occurred while generating image: {e}")
            raise e
        # if response_json['data']:
        #     data = response_json['data'][0]['url']
        #     return ImageOutput(fmt="url", ext="png", data=data)
