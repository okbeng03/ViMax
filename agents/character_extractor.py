import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel, Field
from typing import List
from tenacity import retry, stop_after_attempt
from interfaces import CharacterInScene
from langchain_core.messages import HumanMessage, SystemMessage

from utils.retry import after_func
from utils.completion_logger import log_agent
from utils.provider_presets import create_chat_model
from configs.config import model_name


system_prompt_template_extract_characters = \
"""
[Role]
You are a top-tier movie script analysis expert.

[Task]
Your task is to analyze the provided script and extract all relevant character information.

[Input]
You will receive a script enclosed within <SCRIPT> and </SCRIPT>.

Below is a simple example of the input:

<SCRIPT>
A young woman sits alone at a table, staring out the window. She takes a sip of her coffee and sighs. The liquid is no longer warm, just a bitter reminder of the time that has passed. Outside, the world moves in a blur of hurried footsteps and distant car horns, but inside the quiet café, time feels thick and heavy.
Her finger traces the rim of the ceramic mug, following the imperfect circle over and over. The decision she had to make was supposed to be simple—a mere checkbox on the form of her life. Yes or No. Stay or Go. Yet, it had rooted itself in her chest, a tangled knot of fear and longing.
</SCRIPT>

[Output]
{format_instructions}


[Guidelines]
- Ensure that the language of all output values(not include keys) matches that used in the script.
- Group all names referring to the same entity under one character. Select the most appropriate name as the character's identifier. If the person is a real famous person, the real person's name should be retained (e.g., Elon Musk, Bill Gates)
- If the character's name is not mentioned, you can use reasonable pronouns to refer to them, including using their occupation or notable physical traits. For example, "the young woman" or "the barista".
- For background characters in the script, you do not need to consider them as individual characters.
- If a character's traits are not described or only partially outlined in the script, you need to design plausible features based on the context to make their characteristics more complete and detailed, ensuring they are vivid and evocative.
- In static features, you need to describe the character's physical appearance, physique, and other relatively unchanging features. In dynamic features, you need to describe the character's attire, accessories, key items they carry, and other easily changeable features.
- Don't include any information about the character's personality, role, or relationships with others in either static or dynamic features.
- When designing character features, within reasonable limits, different character appearances should be made more distinct from each other.
- The description of characters should be detailed, avoiding the use of abstract terms. Instead, employ descriptions that can be visualized—such as specific clothing colors and concrete physical traits (e.g., large eyes, a high nose bridge).
"""

# [Role] 
# 你是一流的电影剧本分析专家。 
 
# (任务) 
# 您的任务是分析提供的脚本并提取所有相关的字符信息。 
 
# (输入) 
# 您将收到一个包含在< script >和</ script >中的脚本。 
 
# 下面是一个简单的输入示例： 
 
# < >脚本 
# 一个年轻女子独自坐在桌旁，凝视着窗外。她喝了一口咖啡，叹了口气。液体不再温暖，只是苦涩地提醒着时光的流逝。外面，世界在急促的脚步声和远处汽车的鸣笛声中移动，但在安静的咖啡馆里，时间感觉厚重而沉重。 
# 她的手指沿着陶瓷杯的边缘，沿着不完美的圆圈一圈又一圈地画着。她必须做出的决定应该很简单——仅仅是她生命表格上的一个复选框。Yesor没有。Stayor走。然而，它已经在她的胸膛里扎下了根，一个交织着恐惧和渴望的结。 
# > < /脚本 
 
# (输出) 
# {format_instructions} 
 
 
# (指南) 
# —确保所有输出值（不包括键）的语言与脚本中使用的语言一致。 
# —将指代同一实体的所有名称组合在一个字符下。选择最合适的名称作为字符的标识符。如果这个人是一个真正的名人，应该保留他的真名（例如，埃隆·马斯克，比尔·盖茨）。 
# -如果角色的名字没有被提及，你可以使用合理的代词来指代他们，包括使用他们的职业或显著的身体特征。例如，“年轻的女人”或“咖啡师”。 
# —对于脚本中的背景角色，不需要将其视为单个角色。 
# -如果剧本中没有描述或只是部分概述角色的特征，你需要根据上下文设计合理的特征，使他们的特征更完整和详细，确保他们生动和唤起。 
# -在静态特征中，你需要描述角色的外表，体格和其他相对不变的特征。在动态特征中，你需要描述角色的服装、配饰、他们携带的关键道具以及其他容易改变的特征。 
# -不要在静态或动态特征中包含任何关于角色个性，角色或与其他人关系的信息。 
# -在设计角色特征时，在合理的范围内，使不同的角色外观更加明显。 
# -人物的描述要详细，避免使用抽象的术语。取而代之的是，使用可以可视化的描述——比如特定的衣服颜色和具体的身体特征（例如，大眼睛，高鼻梁）。

human_prompt_template_extract_characters = \
"""
<SCRIPT>
{script}
</SCRIPT>
"""


class ExtractCharactersResponse(BaseModel):
    characters: List[CharacterInScene] = Field(
        ..., description="A list of characters extracted from the script."
    )



class CharacterExtractor:
    def __init__(
        self,
        chat_model,
    ):
        self.chat_model = create_chat_model(
            model_provider="qwen",
            model=model_name.get("secondary", "deepseek-v4-flash-0731"),
        )

    @log_agent("CharacterExtractor")
    @retry(
        stop=stop_after_attempt(3),
        after=after_func,
    )
    async def extract_characters(self, script: str) -> List[CharacterInScene]:

        parser = PydanticOutputParser(pydantic_object=ExtractCharactersResponse)
        
        messages = [
            SystemMessage(content=system_prompt_template_extract_characters.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=human_prompt_template_extract_characters.format(script=script)),
        ]

        chain = self.chat_model | parser

        response: ExtractCharactersResponse = await chain.ainvoke(messages)

        return response.characters

