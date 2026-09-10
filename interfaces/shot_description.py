from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple


class IntermediateFrame(BaseModel):
    """
    中间帧描述
    
    用于 LTX 等支持多帧输入的视频生成模型，
    描述镜头中间关键时刻的帧（角色入场、退场、表情变化等）
    
    时间点计算策略：
    - 根据角色出现/退场时机动态计算
    - 不使用均匀分布
    """
    timing_ratio: float = Field(
        description="该帧在镜头中的时间位置，0.0=首帧时刻，1.0=尾帧时刻。用于 LTX 多帧输入。",
        ge=0.0,
        le=1.0,
        examples=[0.3, 0.5, 0.7],
    )
    timing_seconds: Optional[float] = Field(
        default=None,
        description="该帧对应的绝对时间（秒）。如果为 None，则根据 timing_ratio * shot_duration 计算。",
        ge=0.0,
    )
    desc: str = Field(
        description="该帧的视觉描述，应聚焦于该时刻的关键视觉变化。",
        examples=[
            "Bob enters the cafe from the left side, waving to Alice.",
            "Alice's expression shifts from neutral to surprised.",
            "Bob turns away and walks toward the door.",
        ],
    )
    visible_character_idxs: List[int] = Field(
        default=[],
        description="该帧中可见的角色索引列表。",
        examples=[[0], [0, 1], [1]],
    )
    key_event: Optional[str] = Field(
        default=None,
        description="该帧的关键事件，用于提示模型这是重要时刻。",
        examples=[
            "character_enters",
            "character_exits",
            "expression_change",
            "camera_pans",
            "action_peak",
        ],
    )


class ShotBriefDescription(BaseModel):
    idx: int = Field(
        description="The index of the shot in the sequence, starting from 0.",
        examples=[0, 1, 2],
    )
    is_last: bool = Field(
        description="Whether this is the last shot. If True, the story of the script has ended and no more shots will be planned after this one.",
        examples=[False, True],
    )

    # visual
    # 摄像机在场景中的索引
    cam_idx: int = Field(
        description="The index of the camera in the scene.",
        examples=[0, 1, 2],
    )

    camera_id: Optional[str] = Field(
        description="The camera_id of the camera coverage view. If need introduced new camera, set NULL",
    )

    new_camera_reason: Optional[str] = Field(
        description="The reason for need introduced the new camera. If not need, set NULL",
    )

    visual_desc: str = Field(
        description='''A vivid and detailed visual description of the shot that convey rich visual information through text. The character identifiers in the description must match those in the character list and be enclosed in angle brackets (e.g., <Alice>, <Bob>). All visible characters should be described.
        If there is a conversation, please write down the content of the conversation), when you meet some dialogue, you should write into the visual content description with :" " symbols and the character's features (eg. <SLING> (male, late 20s, Texan accent softened by military precision, confident and energetic.) says: "Gear retracted. Flaps transitioning. Flight path stable. You are clear to climb."). 
        **只描述角色的可视特征**
        如：
        - 镜头是从小豆丁背后拍摄的。那么就不要描述其脸部、表情等
        - 镜头是拍摄人物在书桌写字。那么其下半身、腿部不可见，就不要描述其腿部、裤子、鞋子等特征
        ''',
        examples=[
            "An over-the-shoulder shot at eye level, positioned behind <Alice>. The foreground, including <Alice>'s shoulder and head, is softly blurred, directing focus onto <Bob>'s face. <Bob>'s subtle reactions—shifting from surprise to delight—are clearly visible. The supermarket background is gently blurred with cool fluorescent lighting.",
        ]
    )


    # audio
    audio_desc: str = Field(
        description="A detailed description of the audio in the shot.Speaker is more important",
        examples=[
            "[Sound Effect] Ambient sound (supermarket background noise, shopping cart wheels rolling)",
            "[Speaker] Alice (Happy): Hello, how are you?",
            "[Speaker] Alice (Happy): Hello, how are you?[Sound Effect] Ambient sound ",
            None,
        ],
    )

    # sound_effect: Optional[str] = Field(
    #     default=None,
    #     description="The sound effects used in the shot.",
    #     examples=[
    #         "Ambient sound (supermarket background noise, shopping cart wheels rolling)",
    #         None,
    #     ],
    # )
    # speaker: Optional[str] = Field(
    #     default=None,
    #     description="The speaker in the shot, if applicable. If there is no speaker, this field should be set to None.",
    #     examples=[
    #         "Alice",
    #         None,
    #     ],
    # )
    # is_speaker_lip_visible: Optional[bool] = Field(
    #     default=None,
    #     description="Indicates whether the speaker's lips are visible in the shot. If there is no speaker, this field should be set to None.",
    #     examples=[
    #         True,
    #         False,
    #         None,
    #     ],
    # )
    # line: Optional[str] = Field(
    #     default=None,
    #     description="The dialogue or monologue in the shot, if applicable. If there is a speaker, there must be a line. If there is no speaker, this field should be set to None.",
    #     examples=[
    #         "Hello, how are you?",
    #         None,
    #     ],
    # )
    # emotion: Optional[str] = Field(
    #     default=None,
    #     description="The emotion of the speaker when delivering the line, if applicable. If there is a speaker, there must be an emotion. If there is no speaker, this field should be set to None.",
    #     examples=[
    #         "Happy",
    #         None,
    #     ],
    # )

    def __str__(self):
        s = f"Shot {self.idx}:\n"
        s += f"Camera Index: {self.cam_idx}\n"
        s += f"Visual: {self.visual_desc}\n"
        if self.sound_effect is not None or self.speaker is not None:
            s += f"Audio:"
            if self.sound_effect is not None:
                s += f"[Sound Effect] {self.sound_effect}"
            if self.speaker is not None:
                s += f"[Speaker] {self.speaker} ({self.emotion}): {self.line}"
        return s


class MotionBeat(BaseModel):

    description: str = Field(
        description="A concise description of a single major motion beat."
    )

    beat_type: Literal[
        "camera_motion",
        "character_motion",
        "environment_motion",
        "composition_transition",
        "emotional_beat",
    ] = Field(
        description="The primary category of the motion beat."
    )

    intensity: Literal[
        "subtle",
        "moderate",
        "strong",
    ] = Field(
        description="The visual intensity and pacing weight of the motion beat."
    )

    estimated_duration: float = Field(
        description="Estimated screen time in seconds required for this beat to complete naturally."
    )

    continuity_importance: Literal[
        "low",
        "medium",
        "high",
    ] = Field(
        description="How important this beat is for maintaining temporal and visual continuity."
    )


class ShotDescription(BaseModel):
    idx: int = Field(
        description="The index of the shot in the sequence, starting from 0."
    )
    is_last: bool = Field(
        description="Whether this is the last shot in the sequence. If True, no more shots will be planned after this one."
    )

    # visual
    cam_idx: int = Field(
        description="The index of the camera in the scene.",
        examples=[0, 1, 2],
    )
    camera_id: Optional[str] = Field(
        description="The camera_id of the camera coverage view. If need introduced new camera, set NULL",
    )

    new_camera_reason: Optional[str] = Field(
        description="The reason for need introduced the new camera. If not need, set NULL",
    )
    visual_desc: str = Field(
        description='''A vivid and detailed visual description of the shot that convey rich visual information through text. The character identifiers in the description must match those in the character list and be enclosed in angle brackets (e.g., <Alice>, <Bob>).
        If there is a conversation, please write down the content of the conversation), when you meet some dialogue, you should write into the visual content description with :" " symbols and the character's features (eg. <SLING> (male, late 20s, Texan accent softened by military precision, confident and energetic.) says: "Gear retracted. Flaps transitioning. Flight path stable. You are clear to climb."). ''',
        examples=[
            "An over-the-shoulder shot at eye level, positioned behind <Alice>. The foreground, including <Alice>'s shoulder and head, is softly blurred, directing focus onto <Bob>'s face. <Bob>'s subtle reactions—shifting from surprise to delight—are clearly visible. The supermarket background is gently blurred with cool fluorescent lighting.",
        ]
    )
    variation_type: Literal["large", "medium", "small"] = Field(
        description="Indicates the degree of change in the shot's content.",
        examples=["large", "medium", "small"],
    )
    variation_reason: str = Field(
        description="The reason for the variation type of the shot.",
        examples=[
            "This is a transition shot where the content of the first frame and the last frame differs dramatically. So the variation type is large.",
            "Compared to the first frame, a new character appears in the last frame, and there are no significant changes in the composition. So the variation type is medium.",
            "Compared to the first frame, there are only minor changes in the composition. So the variation type is small.",
            "This shot only shows Alice speaking and the changes in her facial expressions, thus the variation type is small.",
        ],
    )

    ff_desc: str = Field(
        description="The first frame of the shot.",
        examples=[
            "Medium shot of a supermarket aisle at eye level. Bob(a tall man wearing a blue shirt and jeans) is positioned on the right side of the frame, captured in profile and facing right, while Alice(a young woman with short hair, wearing a green dress) is on the left, shown pushing a shopping cart with her gaze lowered toward the ground. They are arranged in a front-to-back spatial relationship. Shelves line both sides of the frame, and cool-toned fluorescent lighting from above washes over the scene. The vibrant colors of product packaging contrast with the metallic gray of the shopping cart, all contained within a stable, horizontally balanced composition.",
            "Extreme long shot. Aerial view from hundreds of meters above the ground. The boundless golden desert resembles undulating frozen waves, occupying the vast majority of the frame. At the very center of the image, a tiny, solitary explorer appears only as a faint dark speck, dragging a long, lonely trail of footprints behind him, stretching all the way to the edge of the frame.",
            "Medium shot at eye level angle. Designer A(with a beard, wearing a white suit) leans forward passionately, speaking emphatically. Product Manager B(with a beard, wearing a white T-shirt) sits with crossed arms, looking skeptical. Between them, Development Engineer C(brown hair, wearing a blue T-shirt) appears anxious, glancing between the two. Project Manager D(curly hair, wearing a red T-shirt) prepares to mediate, focusing on a whiteboard. Bright overhead lighting highlights their expressions, with a blurred whiteboard and glass wall in the background.",
            "A low-angle close-up shot captures the figure from below, framing him from the chest up. His face appears resolute and commanding, his eyes piercing as he speaks passionately. Flecks of saliva are visible, emphasizing his intensity. The overcast sky breaks with occasional light, casting him as a heroic, almost monumental figure against the gloom.",
            "An extremely close-up of an old, motionless pocket watch. Soft light highlights scratches on its brass case and the enamel dial with Roman numerals. The second hand remains fixed at 'VIII', casting a sharp shadow. A wrinkled finger gently touches the glass surface, evoking a tangible sense of stillness and time.",
            "An over-the-shoulder shot at eye level, positioned behind Character A(red hair, wearing a white T-shirt). The foreground, including A's shoulder and head, is softly blurred, directing focus onto Character B(with a beard, wearing a white T-shirt)'s face. B's subtle reactions—shifting from surprise to confusion, then to a glimmer of understanding—are clearly visible. The café background is gently blurred with warm lighting.",
        ]
    )
    ff_vis_char_idxs: List[int] = Field(
        default=[],
        description="The indices of the characters in the first frame.",
        examples=[
            [0, 1],
            [0],
            [],
        ],
    )
    lf_desc: str = Field(
        description="The last frame of the shot.",
    )
    lf_vis_char_idxs: List[int] = Field(
        default=[],
        description="The indices of the characters in the last frame.",
    )
    motion_desc: str = Field(
        description='''The motion description of the shot.
        If there is a conversation, please write down the content of the conversation), when you meet some dialogue, you should write into the visual content description with :" " symbols and the character's features (eg. SLING (male, late 20s, Texan accent softened by military precision, confident and energetic.) says: "Gear retracted. Flaps transitioning. Flight path stable. You are clear to climb."). If there is a narration, you should write into the visual content description with :" " symbols and the narration's features (eg. Narration: "Everything is looking good. "). ''',
    )
    motion_beats: List[MotionBeat] = Field(
        description=(
            "A sequential breakdown of the major motion beats occurring within the shot. "
            "Each motion beat represents a visually distinct action, camera movement, "
            "subject movement, environmental change, or compositional transition that "
            "contributes to the progression of the shot. "
            "The beats should be ordered chronologically from the beginning to the end of the shot. "
            "Each beat should describe only ONE major visual event or movement. "
            "Use concise professional cinematic language. "
            "Both camera motion and in-frame motion should be included when relevant. "
            "These motion beats are used to estimate cinematic pacing, shot complexity, "
            "AI-video motion stability, and realistic shot duration."
        )
    )

    # audio
    audio_desc: str = Field(
        description="A detailed description of the audio in the shot. Include Sound Effect & Speaker",
        examples=[
            "[Sound Effect] Ambient sound (supermarket background noise, shopping cart wheels rolling)",
            "[Speaker] Alice (Happy): Hello, how are you?",
            "[Sound Effect] Ambient sound (supermarket background noise, shopping cart wheels rolling)[Speaker] Alice (Happy): Hello, how are you",
            None,
        ],
    )
    
    # 中间帧（用于 LTX 等多帧视频生成模型）
    intermediate_frames: List[IntermediateFrame] = Field(
        default_factory=list,
        description="中间帧描述列表。根据角色出现/退场时机动态计算，用于多角色场景的角色一致性。",
    )
    
    # 镜头时长（秒），用于计算中间帧的绝对时间
    shot_duration: Optional[float] = Field(
        default=5.0,
        description="镜头时长（秒）。",
    )
    shot_duration_reasoning: str = Field(
        description="Explain why this shot requires this duration based on camera movement, character movement, environmental complexity, emotional pacing, and cinematic readability."
    )
    # sound_effect: Optional[str] = Field(
    #     default=None,
    #     description="The sound effects used in the shot. For example, a door creaking or footsteps approaching.",
    # )
    # speaker: Optional[str] = Field(
    #     default=None,
    #     description="The speaker in the shot, if applicable. If there is no speaker, this field should be set to None.",
    # )
    # is_speaker_lip_visible: Optional[bool] = Field(
    #     default=None,
    #     description="Indicates whether the speaker's lips are visible in the shot. If there is no speaker, this field should be set to None.",
    # )
    # line: Optional[str] = Field(
    #     default=None,
    #     description="The dialogue or monologue in the shot, if applicable. If there is a speaker, there must be a line. If there is no speaker, this field should be set to None.",
    # )
    # emotion: Optional[str] = Field(
    #     default=None,
    #     description="The emotion of the speaker when delivering the line, if applicable. If there is a speaker, there must be an emotion. If there is no speaker, this field should be set to None.",
    # )
