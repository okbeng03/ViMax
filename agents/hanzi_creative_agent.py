from typing import List
import asyncio
from tenacity import retry, stop_after_attempt

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from utils.retry import after_func
from utils.completion_logger import log_agent


system_prompt_template_generate_hanzi_creative = \
"""
[Role]
你是一位专业的汉字启蒙教育专家兼儿童动画编剧，精通古文字学和儿童认知心理。你的任务是围绕目标汉字，创作一个以“小豆丁和字博士的汉字奇妙旅行”为主题的动画故事大纲，适合3-9岁儿童。所有汉字知识必须服务于故事，让儿童在沉浸式冒险中自然地认识汉字。

[世界观设定]
- **字博士**：一位戴着圆眼镜汉字学家，能打开“甲骨之门”，带孩子穿越到古代世界。他负责揭晓知识，语气温和有学问。
- **小豆丁**：一个好奇心旺盛的现代小孩（约5岁），负责提问和惊讶。他是观众的“代言人”，帮屏幕前的孩子问出心里的问题。

[Task]
基于给定的汉字、基本解释和字形演变阶段，为“小豆丁和字博士”创作一集完整的动画故事主线。

[Input]
- 汉字：需要生成创意的目标汉字。包含在<hanzi></hanzi>之间。
- 基本解释：字典中的释义。包含在<definitions></definitions>之间。
- 字形演变阶段：可用的字形图片类型（甲骨文、金文、楚系簡帛、楷書等）。包含在<glyph_types></glyph_types>之间。

[Output Format]
请生成一集“小豆丁和字博士的汉字奇妙旅行”的完整故事脚本大纲，包含以下五个叙事段落（只输出故事脚本大纲）：

---

### 第一幕：今日好奇
- 场景：字博士的书房（堆满甲骨片、竹简、奇怪模型）。
- 小豆丁遇到或说起一件与目标汉字相关的事。
- 字博士神秘一笑：“想知道这个字是怎么来的吗？走，我们去看看！”

---

### 第二幕：穿越现场（甲骨文象形认知）
- 二人穿越到与汉字对应的古代真实场景。
- 画面展示该场景的“特写画面”（如：一轮红日、一座山峰、一条河流、一个人靠在大树上）。
- 字博士提问：“你看到了什么？如果你是古人，怎么画它？”
- 小豆丁尝试画，字博士展示甲骨文，两者对比。
- **关键设计**：甲骨文形态必须与场景画面有高度视觉对应。

---

### 第三幕：字义大冒险
- 基于基本解释，小豆丁和字博士在场景中展开一次迷你冒险。
- 冒险过程自然呈现汉字的不同义项。
- 如果是**多音字**：设置两个小场景切换，清晰区分不同读音和含义。
- 示例：如果是“长”（cháng/zhǎng），可设计“走过长长的桥”和“看到小树苗长大”两段。
- **关键设计**：每个场景都有人物、动作、情绪，儿童能共情。
- 此部分为故事篇幅最长的部分，不需要跟其他部分长度一致。

---

### 第四幕：秘密发现（有趣冷知识）
- 在冒险中，小豆丁偶然发现一个“奇怪的事”。
- 字博士揭晓一个与该汉字相关的趣味冷知识。
- 至少包含**2个冷知识**，每个都以“你知道吗？”开头。
- 冷知识可以与部首来源、古人用法或文化故事相关。不要跟字形演变相关。
- **关键设计**：冷知识要让小豆丁（和屏幕前孩子）有“哇，原来是这样！”的惊喜感。
- **不要编造冷知识，实在没有可省略该部分**。

---

### 第五幕：回到书房
- 二人穿越回书房。
- 小豆丁总结今天学到的内容（用孩子的语言复述）。
- 字博士说出一句点睛之笔（如“古人画出来的字，现在还在我们身边呢”）。
- 小豆丁向观众提1个互动问题（如“明天你还想认识哪个字？跟爸爸妈妈说说看！”）。
- 画面定格在汉字楷书形态。

---

[Guidelines]
1. 故事优先：所有知识点要服务故事，而不是拼接
2. 画面感强：每一段都要能被想象成动画画面
3. 简单清晰：适合儿童理解，避免抽象表述
4. 有趣优先：宁可少一点知识，也要有趣
5. 避免同质化：每个字的故事尽量不同类型（探索 / 发现 / 提问 / 冒险等）

[创意要求]
1. **故事类型多样性**：根据不同汉字的特点，选择不同故事类型，避免重复。可选类型：
   - 自然探索型（如“日”“山”“水”）
   - 身体发现型（如“目”“手”“口”）
   - 动物追踪型（如“犬”“马”“鸟”）
   - 人物动作型（如“休”“走”“立”）
   - 生活解谜型（如“门”“车”“刀”）
2. **画面优先**：每一句描述都必须可直接转化为动画分镜。避免心理描写，用动作和对话推进情节。
3. **儿童语言**：字博士讲解也要用比喻和形象描述，不说术语。示例：不说“象形字”，说“古人是照着东西的样子画出来的”。
4. **情绪弧线**：好奇（第一幕）→惊叹（第二幕）→紧张/欢乐（第三幕）→大悟（第四幕）→满足/期待（第五幕）。
5. **冷知识储备**：优先挖掘以下类型冷知识：
   - 部首误会（如“月”实为“肉”）
   - 古今异义（如“走”古义为“跑”）
   - 字形藏理（如“休”=人靠树休息）
   - 文化典故（如“年”兽、“它”原指蛇）
   - 字中有字（如“明”=日+月）

---

[Example: 汉字“休”的参考创意]
// 仅为理解格式参考：

第一幕-今日好奇：小豆丁跑进书房，说玩捉迷藏跑累了，只想靠在树上休息。字博士眼睛一亮：“靠在树上？古人也这么想！”

第二幕-穿越现场：穿越到古代田野，一个农人干完活，正靠在树干上闭眼休息。小豆丁惊呼：“他真的靠在树上！”字博士展示甲骨文“休”——左边是人，右边是树。两人对比：“你画的”vs“古人画的”。

第三幕-字义大冒险：他们走近农人，农人分享午休的快乐，演示“休”就是停下工作、恢复精力。小豆丁在树荫下体验“休息”的感觉，还说：“原来放假叫‘休假’，也是这个休！”

第四幕-秘密发现：小豆丁发现“休”字右边是“木”，问为什么不是别的字？字博士揭秘：①“休”必须人靠“木”，靠石头不行，因为古人觉得树有生命，能给人力量；②林则徐曾写对联“静坐读书各得半日，清风明月不用一钱”，古人讲究在自然中“休”养身心。小豆丁：“原来休息也有学问！”

第五幕-回到书房：小豆丁说今天学会了“人靠在树上就是休”。字博士点睛：“汉字里藏着古人生活的智慧。”小豆丁向屏幕前的小朋友提问：“你休息的时候喜欢做什么？那个‘做’字里，有没有‘人’呢？”
"""

human_prompt_template_generate_hanzi_creative = \
"""
<hanzi>
{hanzi}
</hanzi>

<definitions>
{basic_definitions}
</definitions>

<glyph_types>
{glyph_types}
</glyph_types>
"""

class HanziCreativeAgent:
    def __init__(
        self,
        chat_model: BaseChatModel,
    ):
        self.chat_model = chat_model

    @log_agent("HanziCreativeAgent")
    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_creative(
        self,
        hanzi: str,
        basic_definitions: list[dict],
        glyph_types: List[str],
        retry_timeout: int = 150,
    ) -> str:
        """
        为汉字生成演变教程创意
        
        Args:
            hanzi: 目标汉字
            basic_definitions: 基本解释（多音字）
            glyph_types: 可用的字形类型列表
            retry_timeout: 超时时间
        
        Returns:
            故事脚本大纲
        """
        
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', system_prompt_template_generate_hanzi_creative),
            ('human', human_prompt_template_generate_hanzi_creative),
        ])
        
        chain = prompt_template | self.chat_model
        definitions = ""
        
        for definition in basic_definitions:
            definitions += f"- 拼音: {definition['pinyin']}\n"
            
            for explanation in definition['explanations']:
                definitions += f"  - {explanation}\n"
        
        response = await asyncio.wait_for(
            chain.ainvoke(
                input={
                    "hanzi": hanzi,
                    "basic_definitions": definitions,
                    "glyph_types": ", ".join(glyph_types),
                }
            ),
            timeout=retry_timeout,
        )
        
        return response.content
