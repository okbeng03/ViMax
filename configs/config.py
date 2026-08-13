"""Standard Template — Balanced quality and speed."""

# Video output settings
VIDEO_WIDTH = 1680
VIDEO_HEIGHT = 960
FPS = 30

working_dir = "/Users/wangchangbin/ai/vimax_men"
evaluate_agent_baseurl = "http://192.168.3.4:8000"
evealute_project_id = "fdf55d15-388f-48e4-a75f-55608ca138a3"

# model_name
model_name = {
    "primary": "qwen3.7-plus",
    "secondary": "qwen3.7-flash-2026-07-15",
    "tertiary": "qwen3.7-flash-2026-07-15",
}

register_voices = {
    "字博士": "字博士",
    "小豆丁": "小豆丁",
    "旁白": "旁白",
    "default": "男生",
    "Male": "男生",
    "Female": "女生",
}

idea = """
汉字教学
"""
user_requirement = """
观众是儿童(3到8岁)。哪吒电影类中国动画，漫画式夸张，热血与幽默并存。故事节奏、推进要合理。
"""
style = "中国动画，国潮美学，东方奇幻，传统纹样，漫画式夸张，硬朗线条，光影戏剧，热血与幽默并存"
