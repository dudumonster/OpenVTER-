#!/usr/bin/env python3
"""VLM prompt templates for VisDrone VRU pseudo OBB v2.

Each prompt template targets a specific VisDrone VRU class and is designed for
cropped images (50-300px wide). Prompts are available in English (for open-source
Grounding models) and Chinese (for Qwen/Gemini).

Usage:
    from dataset_construction.scripts.vlm_prompt_templates import (
        get_grounding_prompt,
        get_review_prompt,
        get_confusion_prompt,
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Grounding prompts (short, descriptive; for GroundingDINO / Florence-2)
# These should be < 80 tokens and describe only visual features.
# ---------------------------------------------------------------------------

GROUNDING_BIKE_EN = (
    "bicycle. bicycle with two wheels. bicycle with frame and pedals."
)
GROUNDING_MOTOR_EN = (
    "motorcycle. motorbike. scooter. electric bike with motor. "
    "two-wheeled motor vehicle."
)
GROUNDING_TRICYCLE_EN = (
    "tricycle. three-wheeled vehicle. three-wheeler with open cargo area."
)
GROUNDING_AWNING_EN = (
    "awning tricycle. covered three-wheeler. "
    "three-wheeled vehicle with canopy. tricycle with roof."
)

# Chinese grounding prompts (for Qwen / Gemini Chinese mode)
GROUNDING_BIKE_ZH = (
    "自行车。两轮自行车。有车架和脚踏的自行车。"
)
GROUNDING_MOTOR_ZH = (
    "摩托车。电动车。两轮机动车辆。有发动机或电池的摩托车。"
)
GROUNDING_TRICYCLE_ZH = (
    "三轮车。三个轮子的车。开放货斗的三轮车。"
)
GROUNDING_AWNING_ZH = (
    "带篷三轮车。有棚顶的三轮车。有车厢的三轮车。封闭式三轮车。"
)

# ---------------------------------------------------------------------------
# Detailed VLM review prompts (for Qwen / Gemini semantic review)
# These include distinguishing rules and format instructions.
# ---------------------------------------------------------------------------

def _make_detailed_en(class_name: str, positive_desc: str, discriminate_from: str) -> str:
    return (
        f"Locate and segment the {class_name} in this cropped drone-view image.\n"
        f"{positive_desc}\n"
        "Instructions:\n"
        "- Segment ONLY the target vehicle body. Exclude: the rider/driver, shadows on the ground, road surface, other vehicles, and background.\n"
        f"{discriminate_from}\n"
        "- From a drone's top-down view, focus on the vehicle's shape, wheel count, and structural features.\n"
        "- If the target is NOT a {class_name}, output the correct class name instead.\n"
        "- Output format: class_name: <name>, bbox: [x1, y1, x2, y2]"
    ).format(class_name=class_name)


def _make_detailed_zh(class_name_zh: str, positive_desc: str, discriminate_from: str) -> str:
    return (
        f"定位并分割这张无人机俯视裁剪图中的{class_name_zh}。\n"
        f"{positive_desc}\n"
        "要求：\n"
        f"- 只分割目标车辆本体。不包含：驾驶员/骑手、地面阴影、路面、其他车辆、背景。\n"
        f"{discriminate_from}\n"
        "- 从无人机俯视角度，关注车辆形状、轮子数量和结构特征。\n"
        f"- 如果目标不是{class_name_zh}，请输出正确的类别名称。\n"
        "- 输出格式：类别: <名称>, 边界框: [x1, y1, x2, y2]"
    )


REVIEW_PROMPTS = {
    "bicycle": {
        "en": _make_detailed_en(
            "bicycle",
            "A bicycle has two wheels, a thin frame, handlebars, pedals, and a chain. "
            "The wheels and thin frame are the most visible features from a drone view.",
            "Discriminate from motorcycles: bicycles have pedals and a chain, no engine/battery housing, no exhaust pipe. "
            "Discriminate from tricycles: bicycles have exactly two wheels, not three.",
        ),
        "zh": _make_detailed_zh(
            "自行车",
            "自行车有两个轮子、细车架、车把、脚踏和链条。从无人机俯视角度看，轮子和细车架是最明显的特征。",
            "与摩托车/电动车区分：自行车有脚踏和链条，没有发动机/电池仓外壳，没有排气管。"
            "与三轮车区分：自行车只有两个轮子，不是三个。",
        ),
    },
    "motor": {
        "en": _make_detailed_en(
            "motorcycle/motorbike/scooter/electric two-wheeler",
            "A motor two-wheeler has two wheels, an engine or battery compartment, a seat, and handlebars. "
            "It may have a rearview mirror and exhaust pipe. No pedals. "
            "The engine/battery housing is a key feature from a drone view.",
            "Discriminate from bicycles: motor vehicles have an engine/battery housing, no pedals, no chain. "
            "Discriminate from tricycles: motor vehicles have exactly two wheels, not three. "
            "Discriminate from cars: motor vehicles are much smaller and narrower.",
        ),
        "zh": _make_detailed_zh(
            "摩托车/电动车/两轮机动车辆",
            "两轮机动车有两个轮子、发动机或电池仓、座椅、车把，可能有后视镜和排气管。没有脚踏。"
            "从无人机俯视角度看，发动机/电池仓外壳是关键特征。",
            "与自行车区分：机动车有发动机/电池仓外壳，没有脚踏和链条。"
            "与三轮车区分：机动车只有两个轮子，不是三个。"
            "与小汽车区分：机动车小得多且窄得多。",
        ),
    },
    "tricycle": {
        "en": _make_detailed_en(
            "tricycle (open three-wheeled vehicle)",
            "A tricycle has THREE wheels, an open cargo area or flatbed at the rear, "
            "and the driver area is open/exposed. No roof, no canopy, no closed cabin. "
            "The three-wheel layout and open rear cargo area are the key features from a drone view.",
            "Discriminate from awning-tricycle: open tricycles have NO roof/awning/canopy, "
            "the cargo area is exposed and visible from above. "
            "If you see a closed or semi-closed cabin/roof structure, it is an awning-tricycle, not a tricycle. "
            "Discriminate from motorcycles: tricycles have three wheels, motorcycles have two.",
        ),
        "zh": _make_detailed_zh(
            "三轮车（开放货斗式三轮车）",
            "三轮车有三个轮子，后面是开放的货斗或平板，驾驶员区域是开放/暴露的。没有棚顶、没有车厢、没有封闭结构。"
            "从无人机俯视角度看，三轮布局和开放的尾部货斗是关键特征。",
            "与带篷三轮车区分：开放三轮车没有棚顶/车厢，货斗区域从上方可见且暴露。"
            "如果看到封闭或半封闭的车厢/棚顶结构，那是带篷三轮车，不是三轮车。"
            "与摩托车区分：三轮车有三个轮子，摩托车只有两个。",
        ),
    },
    "awning_tricycle": {
        "en": _make_detailed_en(
            "awning-tricycle (covered/canopied three-wheeled vehicle)",
            "An awning-tricycle has THREE wheels and a closed or semi-closed cabin/roof/awning/canopy "
            "covering the rear or the entire vehicle. The roof makes it look more like a small boxy vehicle "
            "from above. The covered rectangular body and three-wheel layout are the key features from a drone view.",
            "Discriminate from open tricycle: awning-tricycles HAVE a roof/canopy/cabin structure, "
            "the cargo/passenger area is covered, not open. "
            "Discriminate from small cars/vans: awning-tricycles have three wheels (visible from drone view), "
            "are narrower, and typically have a distinct three-wheel chassis layout. "
            "Discriminate from trucks: awning-tricycles are much smaller.",
        ),
        "zh": _make_detailed_zh(
            "带篷三轮车（有棚顶/车厢的三轮车辆）",
            "带篷三轮车有三个轮子，后部或整体有封闭/半封闭的车厢、棚顶或遮篷结构。"
            "从无人机俯视角度看，封闭的矩形车身和三轮布局是关键特征。",
            "与开放三轮车区分：带篷三轮车有棚顶/车厢结构，货斗/乘客区域是封闭的，不是开放的。"
            "与小汽车/面包车区分：带篷三轮车有三个轮子（从无人机俯视可见）、更窄，通常有明显的三轮底盘布局。"
            "与卡车区分：带篷三轮车小得多。",
        ),
    },
}

# ---------------------------------------------------------------------------
# Confusion / exclusion prompt (ambiguous samples)
# ---------------------------------------------------------------------------

CONFUSION_PROMPT_EN = (
    "This cropped drone-view image may contain one of: bicycle, motorcycle/scooter, "
    "tricycle (open), awning-tricycle (covered), car, van, truck, bus, pedestrian, "
    "or background only.\n"
    "Your task:\n"
    "1. Identify which (if any) vehicle is the MAIN subject centered in this crop.\n"
    "2. If the main subject is bicycle/motorcycle/tricycle/awning-tricycle, segment it precisely.\n"
    "3. If the main subject is a car/van/truck/bus, output 'motor_vehicle' (do not segment).\n"
    "4. If the main subject is a pedestrian or people, output 'pedestrian' (do not segment).\n"
    "5. If the crop is empty or contains only road/background, output 'empty'.\n"
    "Only segment the target vehicle body. Exclude shadows, road markings, people, "
    "and other vehicles."
)

CONFUSION_PROMPT_ZH = (
    "这张无人机俯视裁剪图可能包含以下之一：自行车、摩托车/电动车、三轮车（开放）、"
    "带篷三轮车（有棚顶）、小汽车、面包车、卡车、公交车、行人、或仅背景。\n"
    "你的任务：\n"
    "1. 识别裁剪图中心的主要目标是什么（如果有的话）。\n"
    "2. 如果主要目标是自行车/摩托车/三轮车/带篷三轮车，精准分割它。\n"
    "3. 如果主要目标是汽车/面包车/卡车/公交车，输出'motor_vehicle'（不分割——这些由其他流程处理）。\n"
    "4. 如果主要目标是行人，输出'pedestrian'（不分割）。\n"
    "5. 如果裁剪图是空的或只有路面/背景，输出'empty'。\n"
    "只分割目标车辆本体。排除阴影、路面标线、行人和其他车辆。"
)

# ---------------------------------------------------------------------------
# Prompt selection API
# ---------------------------------------------------------------------------

def get_grounding_prompt(class_name: str, lang: str = "en") -> str:
    """Get the short grounding/detection prompt for a given class.

    Args:
        class_name: One of bicycle, motor, tricycle, awning_tricycle.
        lang: 'en' or 'zh'.

    Returns:
        Short text prompt string suitable for GroundingDINO / Florence-2.
    """
    prompts = {
        "en": {
            "bicycle": GROUNDING_BIKE_EN,
            "motor": GROUNDING_MOTOR_EN,
            "tricycle": GROUNDING_TRICYCLE_EN,
            "awning_tricycle": GROUNDING_AWNING_EN,
        },
        "zh": {
            "bicycle": GROUNDING_BIKE_ZH,
            "motor": GROUNDING_MOTOR_ZH,
            "tricycle": GROUNDING_TRICYCLE_ZH,
            "awning_tricycle": GROUNDING_AWNING_ZH,
        },
    }
    return prompts.get(lang, prompts["en"]).get(class_name, class_name)


def get_review_prompt(class_name: str, lang: str = "en") -> str:
    """Get the detailed VLM review prompt for semantic verification.

    Args:
        class_name: One of bicycle, motor, tricycle, awning_tricycle.
        lang: 'en' or 'zh'.

    Returns:
        Detailed prompt with distinguishing rules for VLM review.
    """
    entry = REVIEW_PROMPTS.get(class_name, {})
    return entry.get(lang, entry.get("en", ""))


def get_confusion_prompt(lang: str = "en") -> str:
    """Get the confusion/exclusion prompt for ambiguous samples.

    Args:
        lang: 'en' or 'zh'.

    Returns:
        Confusion resolution prompt.
    """
    return CONFUSION_PROMPT_EN if lang == "en" else CONFUSION_PROMPT_ZH


def get_all_prompts(lang: str = "en") -> dict[str, str]:
    """Get all grounding prompts as a dict keyed by class name."""
    return {
        name: get_grounding_prompt(name, lang)
        for name in ("bicycle", "motor", "tricycle", "awning_tricycle")
    }
