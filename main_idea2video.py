import asyncio
from pipelines.idea2video_pipeline import Idea2VideoPipeline
import logging

logging.basicConfig(
    level=logging.INFO,  # 日志级别设为INFO，会输出INFO/WARNING/ERROR/CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式（可选）
    handlers=[logging.StreamHandler()]  # 输出到控制台（默认就是，可省略）
)

# SET YOUR OWN IDEA, USER REQUIREMENT, AND STYLE HERE
idea = \
    """
小豆丁在书房看到了一本发光的古书，里面走出字博士，他们开启了汉字演变教学。
"""
user_requirement = \
    """
观众是儿童(3到8岁)，不超过5分钟。
"""
style = "动漫风格"


async def main():
    pipeline = Idea2VideoPipeline.init_from_config(
        config_path="configs/idea2video.yaml")
    await pipeline(idea=idea, user_requirement=user_requirement, style=style)

if __name__ == "__main__":
    asyncio.run(main())
