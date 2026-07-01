"""
IntermediateFramePlanner: 中间帧规划器

根据镜头描述中的角色出现/退场时机，智能规划中间帧的时间位置

核心功能：
1. 分析镜头中角色的出现、退场、表情变化时机
2. 计算中间帧的时间位置（不使用均匀分布）
3. 生成中间帧的视觉描述

用于 LTX 等支持多帧输入的视频生成模型，确保多角色场景的角色一致性
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from tenacity import retry, stop_after_attempt

from interfaces.shot_description import IntermediateFrame, ShotDescription
from interfaces.character import CharacterInScene
from utils.retry import after_func
from utils.completion_logger import log_agent


logger = logging.getLogger(__name__)


# ============= 中间帧规划提示词 =============

system_prompt_template_plan_intermediate_frames = \
"""
[Role]
You are a professional cinematographer specialized in multi-character shot planning. Your expertise is in analyzing when characters appear, exit, or change expressions during a shot, and determining the optimal keyframes for consistent video generation.

[Task]
Your task is to analyze a shot description and identify key moments that require intermediate frames to ensure character consistency in LTX video generation.

LTX and similar models need explicit intermediate frames when:
1. A new character enters the frame
2. A character exits the frame
3. A character's expression/emotion changes significantly
4. A character's position or pose changes significantly
5. Camera movement causes composition change

[Input]
You will receive:
- Shot brief description (visual_desc, audio_desc with dialogues, motion_desc)
- First frame description (ff_desc) and last frame description (lf_desc)
- Character visibility indices for first frame (ff_vis_char_idxs) and last frame (lf_vis_char_idxs)
- Total shot duration (shot_duration)

[Output]
{format_instructions}

[Guidelines]
- Timing Calculation: Use dialogue timing cues from audio_desc to estimate when events occur.
  For example: "Dialogue 1 at 0-1s, Dialogue 2 at 2-3s" suggests character might enter at 2s.
- Character Enter/Exit: If ff_vis_char_idxs ≠ lf_vis_char_idxs, there must be entry/exit events.
- Expression Changes: Map dialogue emotions to timing:
  - Early dialogue → early expression shown in first frames
  - Later dialogue → expression change in middle/later frames
- Avoid Over-generation: Maximum 4 intermediate frames. Only add when necessary.
- Key Event Types:
  - "character_enters": New character appears
  - "character_exits": Character leaves frame
  - "expression_change": Significant emotion shift
  - "position_change": Major character movement
  - "camera_pan": Camera movement completion
- timing_ratio: 0.0 = start of shot, 1.0 = end of shot
- Each intermediate frame description should focus on what's NEW or CHANGED at that moment.
"""

# [Role]
# 你是一名专业的电影摄影师，擅长多角色镜头规划。你的专长是分析角色在镜头中出现、退出或改变表情，并确定一致视频生成的最佳关键帧。

# (任务)
# 你的任务是分析镜头描述并确定需要中间帧的关键时刻，以确保LTX视频生成中的角色一致性。

# LTX和类似模型在以下情况下需要明确的中间帧：
# 1. 一个新角色进入帧
# 2. 一个字符退出帧
# 3. 角色的表情/情绪会发生显著变化
# 4. 角色的位置或姿势会发生显著变化
# 5. 相机的移动引起构图的变化

# (输入)
# 您将收到：
# -镜头简要描述（visual_desc， audio_desc与对话，motion_desc）
# -第一帧描述（ff_desc）和最后一帧描述（lf_desc）
# -第一帧（ff_vis_char_idxs）和最后一帧（lf_vis_char_idxs）的字符可见性索引
# -总拍摄时长（shot_duration）

# (输出)
# {format_instructions}

# (指南)
# -计时计算：使用来自audio_desc的对话计时线索来估计事件发生的时间。
# 例如：“对话1在0- 15，对话2在2-3”表明角色可能在5秒进入。
# —字符进出：当ff_vis_char_idxs≠lf_vis_char_idxs时，必须有字符进出事件。
# -表情变化：将对话情绪映射到时间；
# -早期对话→在第一帧中显示的早期表情
# -后期对话→中/后期画面表情变化
# -避免过度生成：最多4个中间帧。只在必要时添加。
# —关键事件类型：
# —“character_enters”：显示新字符
# —“character_exits”：角色离开帧
# -“expression_change”：显著的情绪转变
# -“position_change”：主要角色移动
# —“camera_pan”：摄像机移动完成
# - timing_ratio: 0.0 =镜头开始，1.0 =镜头结束
# -每一个中间帧的描述应该集中在什么是新的或改变在那一刻。


human_prompt_template_plan_intermediate_frames = \
"""
<SHOT_INFO>
Shot Index: {shot_idx}
Shot Duration: {shot_duration} seconds

Visual Description:
{visual_desc}

Motion Description:
{motion_desc}

Audio Description:
{audio_desc}

First Frame (t=0s):
{ff_desc}
Visible Characters: {ff_vis_char_idxs}

Last Frame (t={shot_duration}s):
{lf_desc}
Visible Characters: {lf_vis_char_idxs}
</SHOT_INFO>

<CHARACTER_LIST>
{characters_str}
</CHARACTER_LIST>
"""


class IntermediateFramePlan(BaseModel):
    """中间帧规划结果"""
    intermediate_frames: List[IntermediateFrame] = Field(
        description="中间帧列表，按时间顺序排列",
    )
    reasoning: str = Field(
        description="规划理由，说明为什么需要这些中间帧",
    )


class IntermediateFramePlanner:
    """
    中间帧规划器
    
    分析镜头描述，智能规划需要生成中间帧的时刻
    """
    
    def __init__(
        self,
        chat_model=None,
        max_intermediate_frames: int = 4,
    ):
        """
        初始化中间帧规划器
        
        Args:
            chat_model: LLM 模型用于分析
            max_intermediate_frames: 最大中间帧数量
        """
        self.chat_model = chat_model
        self.max_intermediate_frames = max_intermediate_frames
    
    def parse_dialogue_timing(self, audio_desc: str, shot_duration: float) -> List[Dict]:
        """
        从音频描述中解析对话时序信息
        
        简化实现：假设对话均匀分布在整个镜头中
        
        Args:
            audio_desc: 音频描述文本
            shot_duration: 镜头时长（秒）
        
        Returns:
            对话时序列表
        """
        dialogues = []
        lines = audio_desc.strip().split('\n') if audio_desc else []
        
        # 提取对话行
        dialogue_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 匹配对话格式
            if ':' in line and not line.startswith('['):
                dialogue_lines.append(line)
        
        if dialogue_lines:
            # 假设对话均匀分布
            interval = shot_duration / (len(dialogue_lines) + 1)
            for i, dialogue in enumerate(dialogue_lines):
                dialogues.append({
                    'start_time': interval * (i + 1),
                    'text': dialogue,
                })
        
        return dialogues
    
    def infer_events_from_comparison(
        self,
        ff_vis_char_idxs: List[int],
        lf_vis_char_idxs: List[int],
        ff_desc: str,
        lf_desc: str,
        shot_duration: float,
    ) -> List[Dict]:
        """
        通过首尾帧比较推断事件
        
        Args:
            ff_vis_char_idxs: 首帧可见角色索引
            lf_vis_char_idxs: 尾帧可见角色索引
            ff_desc: 首帧描述
            lf_desc: 尾帧描述
            shot_duration: 镜头时长
        
        Returns:
            推断的事件列表
        """
        events = []
        
        ff_set = set(ff_vis_char_idxs)
        lf_set = set(lf_vis_char_idxs)
        
        # 角色入场
        entering = lf_set - ff_set
        for char_idx in entering:
            # 假设在镜头中间偏后入场
            timing_ratio = 0.6 + (0.4 * len(entering) / max(len(lf_set), 1))
            events.append({
                'type': 'character_enters',
                'char_idx': char_idx,
                'timing_ratio': min(timing_ratio, 0.95),
                'timing_seconds': min(timing_ratio * shot_duration, shot_duration - 0.5),
            })
        
        # 角色退场
        exiting = ff_set - lf_set
        for char_idx in exiting:
            # 假设在镜头中后期退场
            timing_ratio = 0.5 + (0.3 * len(exiting) / max(len(ff_set), 1))
            events.append({
                'type': 'character_exits',
                'char_idx': char_idx,
                'timing_ratio': min(timing_ratio, 0.9),
                'timing_seconds': min(timing_ratio * shot_duration, shot_duration - 0.5),
            })
        
        # 检查表情/位置变化（通过关键词）
        emotion_keywords = ['smile', 'frown', 'laugh', 'surprise', 'turn', 'look', 'move']
        if any(kw in lf_desc.lower() for kw in emotion_keywords):
            events.append({
                'type': 'expression_change',
                'timing_ratio': 0.5,
                'timing_seconds': shot_duration * 0.5,
            })
        
        return events
    
    def rule_based_plan(
        self,
        shot_description: ShotDescription,
        characters: List[CharacterInScene],
    ) -> List[IntermediateFrame]:
        """
        基于规则的中间帧规划
        
        适用于简单场景的快速规划
        
        Args:
            shot_description: 镜头描述
            characters: 角色列表
        
        Returns:
            中间帧列表
        """
        intermediate_frames = []
        
        # 解析对话时序
        dialogues = self.parse_dialogue_timing(
            shot_description.audio_desc,
            shot_description.shot_duration,
        )
        
        # 推断事件
        events = self.infer_events_from_comparison(
            shot_description.ff_vis_char_idxs,
            shot_description.lf_vis_char_idxs,
            shot_description.ff_desc,
            shot_description.lf_desc,
            shot_description.shot_duration,
        )
        
        # 合并事件（去重、排序）
        all_events = []
        
        for event in events:
            all_events.append(event)
        
        for dialogue in dialogues:
            all_events.append({
                'type': 'dialogue_moment',
                'timing_ratio': dialogue['start_time'] / shot_description.shot_duration,
                'timing_seconds': dialogue['start_time'],
                'text': dialogue['text'],
            })
        
        # 按时间排序
        all_events.sort(key=lambda x: x['timing_ratio'])
        
        # 限制数量
        all_events = all_events[:self.max_intermediate_frames]
        
        # 生成中间帧描述
        for i, event in enumerate(all_events):
            timing_ratio = event['timing_ratio']
            
            # 生成描述
            if event['type'] == 'character_enters':
                char_idx = event.get('char_idx')
                if char_idx is not None and char_idx < len(characters):
                    char_name = characters[char_idx].identifier_in_scene
                    desc = f"{char_name} enters the scene, approaching their position."
                else:
                    desc = "A character enters the frame."
                key_event = "character_enters"
                visible_idxs = list(set(shot_description.ff_vis_char_idxs + [char_idx]))
                
            elif event['type'] == 'character_exits':
                char_idx = event.get('char_idx')
                desc = f"A character exits the frame, leaving the scene."
                key_event = "character_exits"
                visible_idxs = [c for c in shot_description.lf_vis_char_idxs]
                
            elif event['type'] == 'expression_change':
                desc = "Character expression changes from neutral to emotional."
                key_event = "expression_change"
                visible_idxs = shot_description.ff_vis_char_idxs
                
            elif event['type'] == 'dialogue_moment':
                text = event.get('text', '')
                desc = f"Speaking: {text}"
                key_event = "dialogue"
                visible_idxs = shot_description.ff_vis_char_idxs
                
            else:
                desc = f"Key moment at {timing_ratio * shot_description.shot_duration:.1f}s."
                key_event = "action_peak"
                visible_idxs = shot_description.ff_vis_char_idxs
            
            intermediate_frames.append(IntermediateFrame(
                timing_ratio=timing_ratio,
                timing_seconds=event.get('timing_seconds'),
                desc=desc,
                visible_character_idxs=visible_idxs,
                key_event=key_event,
            ))
        
        return intermediate_frames
    
    @log_agent("IntermediateFramePlanner")
    @retry(stop=stop_after_attempt(3), after=after_func)
    async def llm_plan(
        self,
        shot_description: ShotDescription,
        characters: List[CharacterInScene],
        retry_timeout: int = 60,
    ) -> List[IntermediateFrame]:
        """
        使用 LLM 进行中间帧规划
        
        适用于复杂场景的精确规划
        
        Args:
            shot_description: 镜头描述
            characters: 角色列表
            retry_timeout: 超时时间
        
        Returns:
            中间帧列表
        """
        if not self.chat_model:
            logger.warning("No chat model provided, falling back to rule-based planning")
            return self.rule_based_plan(shot_description, characters)
        
        parser = PydanticOutputParser(pydantic_object=IntermediateFramePlan)
        
        characters_str = "\n".join([
            f"Character {i}: {char.identifier_in_scene} ({char.static_features})"
            for i, char in enumerate(characters)
        ])
        
        messages = [
            SystemMessage(content=system_prompt_template_plan_intermediate_frames.format(
                format_instructions=parser.get_format_instructions()
            )),
            HumanMessage(content=human_prompt_template_plan_intermediate_frames.format(
                shot_idx=shot_description.idx,
                shot_duration=shot_description.shot_duration,
                visual_desc=shot_description.visual_desc,
                motion_desc=shot_description.motion_desc,
                audio_desc=shot_description.audio_desc or "(No audio)",
                ff_desc=shot_description.ff_desc,
                ff_vis_char_idxs=shot_description.ff_vis_char_idxs,
                lf_desc=shot_description.lf_desc,
                lf_vis_char_idxs=shot_description.lf_vis_char_idxs,
                characters_str=characters_str,
            )),
        ]
        
        import asyncio
        chain = self.chat_model | parser
        
        try:
            result: IntermediateFramePlan = await asyncio.wait_for(
                chain.ainvoke(messages),
                timeout=retry_timeout,
            )
            
            logger.info(f"LLM planned {len(result.intermediate_frames)} intermediate frames: {result.reasoning}")
            return result.intermediate_frames
            
        except Exception as e:
            logger.error(f"LLM planning failed: {e}, falling back to rule-based")
            return self.rule_based_plan(shot_description, characters)
    
    async def plan(
        self,
        shot_description: ShotDescription,
        characters: List[CharacterInScene],
    ) -> List[IntermediateFrame]:
        """
        规划中间帧
        
        自动选择规划方式：
        - 简单场景：规则规划
        - 复杂场景：LLM 规划
        
        Args:
            shot_description: 镜头描述
            characters: 角色列表
        
        Returns:
            中间帧列表
        """
        # 判断是否需要中间帧
        ff_set = set(shot_description.ff_vis_char_idxs)
        lf_set = set(shot_description.lf_vis_char_idxs)
        
        # 如果首尾帧角色集合不同，需要中间帧
        needs_intermediate = (ff_set != lf_set) or len(characters) > 1
        
        if not needs_intermediate:
            return []
        
        # 选择规划方式
        if self.chat_model and len(characters) > 1:
            return await self.llm_plan(shot_description, characters)
        else:
            return self.rule_based_plan(shot_description, characters)
    
    def plan_batch(
        self,
        shot_descriptions: List[ShotDescription],
        characters: List[CharacterInScene],
    ) -> List[List[IntermediateFrame]]:
        """
        批量规划中间帧
        
        Args:
            shot_descriptions: 镜头描述列表
            characters: 角色列表
        
        Returns:
            中间帧列表的列表
        """
        import asyncio
        
        async def plan_all():
            return [await self.plan(sd, characters) for sd in shot_descriptions]
        
        return asyncio.run(plan_all())


# 便捷函数
async def plan_intermediate_frames(
    shot_description: ShotDescription,
    characters: List[CharacterInScene],
    chat_model=None,
) -> List[IntermediateFrame]:
    """
    规划中间帧的便捷函数
    
    Args:
        shot_description: 镜头描述
        characters: 角色列表
        chat_model: 可选的 LLM 模型
    
    Returns:
        中间帧列表
    """
    planner = IntermediateFramePlanner(chat_model=chat_model)
    return await planner.plan(shot_description, characters)


# 示例用法
if __name__ == "__main__":
    # 测试规则规划
    from interfaces.shot_description import ShotDescription, IntermediateFrame
    
    shot = ShotDescription(
        idx=0,
        is_last=False,
        cam_idx=0,
        visual_desc="Medium shot in a cafe. Alice sits by the window, Bob enters and approaches.",
        ff_desc="Alice sits by the window, smiling.",
        ff_vis_char_idxs=[0],
        lf_desc="Bob and Alice are talking, sitting across from each other.",
        lf_vis_char_idxs=[0, 1],
        motion_desc="Static camera. Bob walks into frame and sits down.",
        audio_desc="Alice: 你好 Bob！\nBob: 你好 Alice！",
        shot_duration=5.0,
    )
    
    class MockCharacter:
        def __init__(self, idx, name):
            self.idx = idx
            self.identifier_in_scene = name
            self.static_features = f"age={20+idx*5}, gender=female"
            self.dynamic_features = ""
    
    characters = [MockCharacter(0, "Alice"), MockCharacter(1, "Bob")]
    
    planner = IntermediateFramePlanner()
    frames = planner.rule_based_plan(shot, characters)
    
    print(f"Planned {len(frames)} intermediate frames:")
    for frame in frames:
        print(f"  - t={frame.timing_ratio*5:.1f}s ({frame.timing_ratio:.2f}): {frame.desc}")
