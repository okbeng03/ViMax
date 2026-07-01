import asyncio
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from utils.completion_logger import set_working_dir
import logging
from configs.config import idea, user_requirement, style

logging.basicConfig(
    level=logging.INFO,  # 日志级别设为INFO，会输出INFO/WARNING/ERROR/CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式（可选）
    handlers=[logging.StreamHandler()]  # 输出到控制台（默认就是，可省略）
)

async def main():
    pipeline = Idea2VideoPipeline.init_from_config(
        config_path="configs/idea2video.yaml")
    set_working_dir(pipeline.working_dir)
    await pipeline(idea=idea, user_requirement=user_requirement, style=style)

if __name__ == "__main__":
    asyncio.run(main())
