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
一个穿着有卡皮巴拉图案的泳衣黑色头发6岁中国男孩，和他的动物好朋友卡皮巴拉一起游戏泼水的游玩故事。
"""
user_requirement = \
    """
观众是儿童，3个场景。每个场景应不超过5张画面。
"""
style = "动漫风格"


async def main():
    pipeline = Idea2VideoPipeline.init_from_config(
        config_path="configs/idea2video.yaml")
    await pipeline(idea=idea, user_requirement=user_requirement, style=style)

if __name__ == "__main__":
    asyncio.run(main())
