import asyncio
from pipelines.hanzi_pipeline import HanziPipeline
import logging

logging.basicConfig(
    level=logging.INFO,  # 日志级别设为INFO，会输出INFO/WARNING/ERROR/CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式（可选）
    handlers=[logging.StreamHandler()]  # 输出到控制台（默认就是，可省略）
)


async def main():
    pipeline = HanziPipeline.init_from_config(
        config_path="configs/hanzi.yaml")
    await pipeline()

if __name__ == "__main__":
    asyncio.run(main())
