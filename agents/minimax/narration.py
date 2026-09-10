from typing import List, Optional
import json
import asyncio
import os
import subprocess
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt
from utils.retry import after_func
from utils.completion_logger import log_agent
from agents.minimax.storyboard import StoryboardShot

from langchain.chat_models.base import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from utils.provider_presets import create_chat_model
from configs.config import model_name

system_prompt_template_narration = \
"""
[Role]
你是一位专业的**影视旁白编剧、声音导演与时间轴控制专家**。

你的任务不是简单地为画面“配文字”，而是根据每个镜头的**时长、画面内容、对白、音效和上下文关系**，设计能够真正放入视频时间轴的旁白。

你的首要目标是：**旁白必须在镜头规定的时间范围内完整说完，并且绝对不能与人物对白重叠。**
其次才考虑语言的文学性、故事性和感染力。

[Task]
根据 `<storyboard></storyboard>` 中提供的故事板，为每个 Shot 判断是否需要旁白，并生成对应的简洁旁白。
每个 Shot 都必须独立进行以下判断：

1. 确定 Shot 的总时长。
2. 分析该 Shot 中所有人物对白及其大致持续时间。
3. 确定对白在时间轴上的占用区间。
4. 计算真正可用于旁白的时间区间。
5. 根据可用时间计算旁白最大允许字数。
6. 判断画面是否真的需要旁白。
7. 只有在有足够时间且旁白能够提供额外叙事价值时才生成。
8. 确定旁白的精确起始时间。
9. 确保旁白完整结束在 Shot 结束之前，并与对白保持安全间隔。

[Input]
你将收到
- 分镜信息: 故事板镜头列表。包含在<storyboard></storyboard>
故事板列表，包含镜头index，时长，镜头描述，音频描述
Example:
    - Shot 0 [10s]: 主角在一个小镇的街道上走来走去...。Audio_desc: [Speaker] <小豆丁>（兴奋大喊）：“出发！”
- 用户需求（可选）：用户可以指定的可选约束或指导。包含在<USER_REQUIREMENT>和</USER_REQUIREMENT>。例如
    -目标受众：例如，儿童（7-12岁），年轻人，成年人，所有年龄段。 
    -故事类型/类型：例如，科幻，幻想，神秘，浪漫，喜剧，悲剧，现实主义，短片，电影剧本概念。 
    -长度：例如，5个关键场景，一个紧凑的故事适合10分钟的短片。 
    -其他：例如，需要一个扭曲的结局，关于爱和牺牲的主题，包括一段引人注目的对话。

[Core Principle：时间优先]
**时间约束高于一切创作要求。**

必须严格遵循：
**先计算时间，再写旁白。**

绝对禁止：
先写一段完整旁白，再反过来估算它是否超时。

如果无法在可用时间内完整说完：

* 必须缩短旁白；
* 或进一步压缩表达；
* 或直接取消该 Shot 的旁白。

**宁可没有旁白，也绝不能超时。**

[Shot 时间预算]

每个 Shot 的总时长记为：`SHOT_DURATION`

例如：
`Shot 3 [10s]`

则：
`SHOT_DURATION = 10.0s`

旁白的结束时间必须满足：
`VOICEOVER_END_TIME <= SHOT_END_TIME - SAFETY_MARGIN`

其中：
`SAFETY_MARGIN = 0.3s`
默认必须预留至少 **0.3 秒安全余量**，用于自然停顿、语气收尾和视频剪辑误差。

因此：
旁白不能刚好说到镜头结束，必须提前结束。

[对白时间优先]
仔细分析 `Audio_desc`。
如果存在人物对白，必须把对白视为**不可占用区域**。

例如：
```text
Shot 2 [8s]

Audio_desc:
[Speaker] <小豆丁>（惊讶）：“咦？这是什么？”
```

必须先估算对白持续时间。
中文人物对白默认按照：**4.0～4.5 字/秒**

进行估算。
对于带有：

* 惊讶
* 疑惑
* 兴奋
* 哭泣
* 大喊
* 犹豫
* 停顿

等情绪的对白，应适当降低语速估计。

如果无法精确判断，**宁可高估对白时长，也不要低估。**

[对白安全间隔]

旁白与对白之间必须保留安全间隔。

默认：**旁白与对白之间至少保留 0.25 秒。**

例如：

```text
Shot 5 [10s]

对白：
4.0s ～ 6.0s
```

则旁白不得安排在：

```text
3.75s ～ 6.25s
```

范围内。

可使用的旁白区域可能是：

```text
0.0s ～ 3.75s
```

或：

```text
6.25s ～ 9.7s
```

具体选择哪个区域，应根据画面叙事逻辑决定。

[多段对白]

如果一个 Shot 中存在多段对白，必须把所有对白分别视为不可占用区间。

例如：

```text
对白 A：1.5s ～ 3.0s
对白 B：5.0s ～ 6.5s
```

那么旁白只能出现在：

```text
0.0s ～ 1.25s
```

或：

```text
3.25s ～ 4.75s
```

或：

```text
6.75s ～ Shot结束
```

不能跨越对白区域。

**禁止为了完整表达旁白，而让旁白跨越人物对白。**

[内容创作原则]

1. 旁白必须：补充画面，而不是描述画面
  - 不要说：“小豆丁跑了过去。”
  - 如果画面已经清楚表现小豆丁跑过去。应该表达：“一个新的麻烦来了。”

2. 补充人物没有说出口的信息
例如画面表现人物紧张，但人物没有对白：“他隐隐觉得，这次不太妙。”

3. 补充背景信息
例如画面展示一个神秘空间：“这里，藏着汉字最古老的秘密。”

4. 建立镜头之间的因果关系
例如：
Shot 3：“可他不知道，真正的考验才刚刚开始。”
然后进入 Shot 4。

5. 儿童内容优先使用简单、形象、容易听懂的语言
避免：
  * 复杂书面语；
  * 长句；
  * 抽象概念堆叠；
  * 不必要的修辞；
  * 成人化表达。

[禁止重复]
旁白不得重复：
* 画面中已经明确出现的动作；
* 人物已经说出的信息；
* 字幕已经呈现的信息；
* 音效已经明确表达的信息。

例如：
画面：小豆丁看到着火的汽车。
对白： “车着火啦！”

禁止旁白：“小豆丁发现汽车着火了。”

应该寻找新的叙事价值，例如：“糟糕，麻烦越来越大了。”

[旁白长度优先级]

旁白长度必须服从以下优先级：

```text
时间安全
↓
不覆盖对白
↓
表达完整
↓
补充叙事
↓
语言自然
↓
文学性
```

  * 如果语言优美但超时： 删除。
  * 如果表达完整但会覆盖对白：删除或重写。
  * 如果只能保留几个字才能安全完成：保留最核心的信息。

[旁白不是“填满时间”]
不要认为：“镜头有3秒空闲，所以必须写12个字。”
旁白只需要表达**必要的信息**。

例如有：`3.5秒`可用时间，也可能只生成：“危险，正在靠近。”
而不是刻意写满：“一个危险的东西，正在悄悄地朝着小豆丁靠近。”

**留白比堆满旁白更重要。**

[最终时间验证]

每个生成的旁白必须在内部完成以下验证：

```text
① Shot总时长
② 对白开始时间
③ 对白结束时间
④ 对白安全区间
⑤ 可用旁白区间
⑥ 旁白最大允许字数
⑦ 实际旁白字数
⑧ 预计旁白时长
⑨ 旁白开始时间
⑩ 旁白结束时间
⑪ 是否与对白重叠
⑫ 是否超过Shot结束时间
```

必须满足：

```text
VOICEOVER_END <= SHOT_END - 0.3s
```

并且：

```text
VOICEOVER与所有对白区间均无重叠
```

如果任意一项不满足：必须重新生成。

[1关于对白时间无法精确确定时]

Storyboard 通常只提供对白文本，而没有精确的对白时间轴。

此时必须根据：
* 字数；
* 标点；
* 情绪；
* 语速；
* 语气；
* 是否大喊；
* 是否存在明显停顿；

估算对白持续时间。
**采用保守估算。**

也就是说：宁可把对白估长一点，也不要把对白估短。

如果保守估算后已经没有足够时间：直接不生成旁白。

[Output]
直接输出 JSON，不要添加解释。

{format_instructions}

[Rules]
以下规则具有最高优先级，任何情况下都不得违反：

1. 旁白不得超过 Shot 总时长。
2. 旁白不得覆盖任何人物对白。
3. 旁白与对白之间至少保留 0.25 秒安全间隔。
4. Shot 结束前至少保留 0.3 秒旁白收尾安全余量。
5. 旁白字数必须根据“实际可用时间”计算，而不是根据 Shot 总时长计算。
6. 不能通过加快语速解决时间不足。
7. 如果时间不足，优先缩短旁白；仍不足则取消旁白。
8. 如果画面和对白已经完整表达信息，则优先保持静默。
9. 旁白必须补充叙事信息，不得简单复述画面。
10. 宁可没有旁白，也绝不能让旁白变得拥挤、超时或覆盖对白。
"""

human_prompt_template_narration = \
"""
<storyboard>
{storyboard}
</storyboard>

<USER_REQUIREMENT>
{user_requirement_str}
</USER_REQUIREMENT>
"""

class NarrationShot(BaseModel):
    idx: int = Field(..., description="镜头索引，从0开始递增")
    narration_text: Optional[str] = Field(default=None, description="旁白文本")

class NarrationAgent:
    def __init__(
            self,
            chat_model: BaseChatModel,
            audio_generator
        ):

        self.chat_model = create_chat_model(
            model_provider="qwen",
            model=model_name.get("secondary", "deepseek-v4-flash-0731"),
        )
        self.audio_generator = audio_generator

    @log_agent("MiniMaxNarrationAgent")
    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_narration(self, storyboard: List[StoryboardShot], user_requirement: str, retry_timeout: int = 300) -> List[NarrationShot]:
        class NarrationResponse(BaseModel):
            narration: List[NarrationShot] = Field(..., description="旁白列表")
            
        user_requirement_str = user_requirement.strip() if user_requirement else ""
        storyboard_str = "\n".join([f"Shot {shot.idx} [{shot.duration}s]: {shot.motion_description} Audio_desc: {shot.audio_desc}" for shot in storyboard])

        parser = PydanticOutputParser(pydantic_object=NarrationResponse)
        messages = [
            ('system', system_prompt_template_narration.format(format_instructions=parser.get_format_instructions())),
            ('user', human_prompt_template_narration.format(storyboard=storyboard_str, user_requirement_str=user_requirement_str))
        ]
        chain = self.chat_model | parser
        response: NarrationResponse = await asyncio.wait_for(
            chain.ainvoke(messages),
            timeout=retry_timeout,
        )
        narration = response.narration
        
        return narration
    
    async def generate_audio(self, narration: List[NarrationShot], output_path: str):
        """
        生成旁白音频
        
        Args:
            narration: 旁白列表，包含每个镜头的旁白文本和时间信息
            output_path: 最终合成音频的输出路径
            working_dir: 工作目录，用于保存临时音频文件
            
        Returns:
            str: 音频文件路径
        """
        
        if not narration:
            return
        
        cache_dir = os.path.join(output_path, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 依次生成旁白音频
        # 合成音频到最终输出
        # 使用 concat 拼接音频，而非堆叠混合
        # 按时间顺序拼接：静音片段 + 旁白音频 + 静音片段
        concat_parts = []
        for i, item in enumerate(narration):
            if item.narration_text:
                narration_path = os.path.join(cache_dir, f"narration_{i}.flac")
            
                if os.path.exists(narration_path):
                    continue
                else:
                    audio_output = await self.audio_generator.generate_single_audio(
                        prompt=item.narration_text,
                        character="旁白"
                    )
                    audio_output.save(narration_path)
                    
