# https://ai.google.dev/gemini-api/docs/image-generation

import logging
import asyncio
from PIL import Image
from typing import List, Optional
import dashscope
from dashscope import MultiModalConversation
from tenacity import retry, stop_after_attempt
from interfaces.image_output import ImageOutput
from utils.retry import after_func
from utils.rate_limiter import RateLimiter


class ImageGeneratorDashscopeQwenAPI:
    def __init__(
        self,
        api_key: str,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.model = "qwen-image-edit-plus-2025-12-15"
        self.rate_limiter = rate_limiter
        # self.client = genai.Client(
        #     api_key=api_key,
        # )
        self.api_key = api_key

    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        aspect_ratio: Optional[str] = "16:9",
        **kwargs,
    ) -> ImageOutput:

        """
            aspect_ratio: The aspect ratio of the image.
        """

        logging.info(f"Calling {self.model} to generate image...")

        # Apply rate limiting if configured
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        reference_images = [Image.open(path) for path in reference_image_paths]

        # Retry logic for rate limit errors
        max_retries = 3
        retry_delay = 5
        
        content = []
        
        for image in reference_images:
            content.append({
                "image": image,
            })
            
        content.append({
            "text": prompt,
        })
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        
        model = "qwen-image-edit-plus-2025-12-15" if len(reference_images) else "qwen-image-max-2025-12-30"

        for attempt in range(max_retries):
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model=model,
                messages=messages,
                stream=False,
                n=1,
                watermark=False,
                negative_prompt="低质量",
                prompt_extend=True,
                # 仅当输出图像数量n=1时支持设置size参数，否则会报错
                size="1024*1024",
            )
            
            if response.status_code == 200:
                break
            else:
                wait_time = retry_delay * (2 ** attempt)
                logging.warning(f"Generate image error, statusCode: {response.status_code}, code: {response.code}, message: {response.message}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)

        image = response.output.choices[0].message.content[0]["image"]

        if image is None:
            logging.error(f"No image generated. The response text is: {response.output.choices[0].message.content[0]}")
            raise ValueError("No image generated")

        return ImageOutput(fmt="url", ext="png", data=image)

