# VLM 辅助 VisDrone 弱势交通参与者 OBB 伪标注方案 v2

**版本**: v2 (draft)
**日期**: 2026-06-25
**状态**: 方案设计阶段，待 Codex 视觉验证后迭代
**关联方案**: [visdrone_hbb_to_obb.md](visdrone_hbb_to_obb.md) (v1)

## 1. 项目背景

OpenVTER 项目使用无人机航拍交通视频进行车辆检测、跟踪和轨迹分析。现阶段 5 类 OBB 训练目标为：

| OBB 训练类别 ID | 类别名 | 来源 |
| --- | --- | --- |
| 0 | `motor_vehicle` | DroneVehicle / DOTA / VSAI / UAV-OBB 等机动车 OBB 数据集 |
| 1 | `bicycle` | VisDrone class_id=3 |
| 2 | `motor` | VisDrone class_id=10 |
| 3 | `tricycle` | VisDrone class_id=7 |
| 4 | `awning_tricycle` | VisDrone class_id=8 |

弱点：VisDrone 原始标注是 HBB（水平边界框），没有方向信息。v1 方案通过 HBB 裁剪 + SAM/SAM2/GrabCut 前景分割 + cv2.minAreaRect 拟合 OBB，在小目标、遮挡、阴影、多目标粘连等场景下效果不稳定。

## 2. 为什么需要 VLM / open-vocabulary 模型

v1 方案的三个核心缺陷：

1. **分割模型不理解语义**。SAM/SAM2 是通用分割模型，对 "这是自行车还是摩托车" 没有概念。当 HBB 裁剪区域内包含人+车粘连、多车并列、部分遮挡时，它不知道应该分割哪一个目标主体。

2. **分割质量不稳定**。SAM/SAM2 的 box-prompt 模式在 VisDrone 这种 2000×1500 分辨率下 20-60px 的小目标上表现波动大，经常分割出不完整轮廓或包含背景。

3. **无法区分细粒度类别**。tricycle vs awning_tricycle 的区别在于是否有棚顶，motor vs bicycle 的区别在于车身结构。纯分割模型无法利用这些语义信息，只能依赖 VisDrone 原始 HBB 的 class_id 标签继承——但 HBB 标签本身也可能有标注错误。

引入 VLM / open-vocabulary 模型的核心价值：**把大模型的语义理解能力用作弱监督教师**，在目标定位和类别确认两个环节提升伪标注质量。

## 3. 为什么不直接相信 VLM 输出

VLM 输出存在以下不确定性，因此本方案把 VLM 定位为**弱监督伪标注教师，而不是最终真值来源**：

1. **VLM 幻觉**。VLM 可能在空白区域 "想象" 出目标，或者在语义模糊时给出随机输出。
2. **定位精度有限**。VLM 输出的 bounding box 通常不如专用检测器精确，更不如 SAM2 的像素级 mask。
3. **开放词汇的不确定性**。不同 VLM 对同一 prompt 的理解不同，输出格式也不统一。
4. **成本和速度**。大规模调用在线 VLM API（Gemini / Qwen / GPT-4V）成本高、延迟大，不适合做全量标注。

因此本方案的设计哲学是：**VLM + Grounding + SAM2 组合 > 单独 VLM > 单独 SAM2 > 单独 GrabCut**。每一步都有质量检查和人工审核兜底。

## 4. 推荐模型路线

### 主方案：Grounded-SAM-2（或 GroundingDINO + SAM2）

```
VisDrone HBB 裁剪 → GroundingDINO / Florence-2 文本定位 → SAM2 box/mask prompt → mask → minAreaRect OBB
```

- **GroundingDINO**：根据文本 prompt 在裁剪区域内找到目标，输出 box。
- **Florence-2**：微软的轻量视觉基础模型，支持 open-vocabulary detection + phrase grounding，可以作为 GroundingDINO 的替代或补充。
- **SAM2**：根据 Grounding 输出的 box 生成精细 mask。
- **mask → OBB**：使用 cv2.minAreaRect 对最大连通域拟合旋转矩形。

选这条路的原因：
1. Grounding + SAM 的组合在工业界已有大量验证（IDEA-Research 的 Grounded-SAM 系列）。
2. mask 可以通过 minAreaRect 稳定转为 OBB，比直接让 VLM 输出四边形点坐标更可靠。
3. 每一步可独立调试、替换后端。

### 辅助方案

| 方案 | 用途 | 优先级 |
| --- | --- | --- |
| Qwen2.5/3-VL | 语义审核器：对 auto_accept 的样本做二次确认，对 ambiguous 样本做裁决 | 中 |
| Gemini 2.5 Pro | 少量高质量伪标注教师（每类 50-100 个样本，直接输出四边形坐标 + 类别） | 低 |
| YOLO-World | 高速 open-vocabulary 候选框生成器，替代 GroundingDINO 做初筛 | 中 |
| DINO-X | GroundingDINO 的升级替代品 | 低 |
| NVIDIA TAO / CVAT / Label Studio | 后续工程化标注平台方向，不在本轮范围 | 远期 |

### 后端选择矩阵

| 后端组合 | Grounding 模型 | 分割模型 | 适用场景 | 速度 |
| --- | --- | --- | --- | --- |
| `grounded_sam2` | GroundingDINO | SAM2 | 通用，精度最高 | 慢 |
| `groundingdino_sam2` | GroundingDINO (独立部署) | SAM2 | 与上同，独立控制版本 | 慢 |
| `florence2_sam2` | Florence-2 | SAM2 | 轻量、快速 | 中 |
| `yoloworld_sam2` | YOLO-World | SAM2 | 超快速初筛 | 快 |
| `qwen` | Qwen2.5/3-VL (端到端) | — | 直接输出 OBB + 类别 | 中 |
| `gemini` | Gemini (API) | — | 直接输出 OBB + 类别 | 慢（含网络延迟） |
| `wan` | Wan2.7-Image | — | 图像编辑/增强辅助 | — |

## 5. 数据流设计

### 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VisDrone HBB 标注                                │
│   class_id ∈ {3:bicycle, 7:tricycle, 8:awning_tricycle, 10:motor}  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: 裁剪 + 外扩                                                │
│   expand_ratio = 0.20 (v2 增大到 20%，给 VLM/Grounding 更多上下文) │
│   保留裁剪区域的全局坐标映射                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: VLM / Grounding 辅助定位                                    │
│   - 输入裁剪图 + 文本 prompt                                        │
│   - GroundingDINO/Florence-2 输出 refined box                       │
│   - （可选）Qwen/Gemini 输出语义类别确认                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: SAM2 mask 生成                                              │
│   - 输入原图 + refined box（或 HBB expanded box）                   │
│   - SAM2 输出 mask                                                  │
│   - 选择最佳连通域（离 HBB 中心最近的大连通域）                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: mask → OBB                                                  │
│   - cv2.minAreaRect 拟合                                            │
│   - 坐标还原到原图                                                   │
│   - 四点顺时针排序                                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: 质量评分                                                    │
│   - 几何指标（area_ratio, center_shift, foreground_ratio, etc.）   │
│   - 语义指标（class_confidence from VLM）                           │
│   - 综合评分                                                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        高质量样本       中等质量样本      低质量样本
      (auto_accept)     (review)        (reject)
              │              │              │
              ▼              ▼              ▼
        直接入库        人工审核队列      记录失败原因
                        │                 │
                        ▼                 ▼
                   审核后入库          排除/重新标注
```

### 与 v1 流水线的关系

v2 不替代 v1，而是作为 v1 的增强版：

- v1 流水线保持不变，路径为 `visdrone_pseudo_obb_v1` / `visdrone_yolo_obb_v1`。
- v2 新增独立路径 `visdrone_pseudo_obb_v2` / `visdrone_yolo_obb_v2`。
- v2 复用的部分：VisDrone 数据读取、annotation 解析、expand_box、select_component、obb_from_mask、quality scoring 框架、review 审核网页、apply_review 应用逻辑。

## 6. 类别映射

从 VisDrone 到 5 类 OBB 训练类别的映射（与 v1 相同）：

| VisDrone class_id | VisDrone 类别 | OBB 训练 class_id | OBB 训练类别 |
| --- | --- | --- | --- |
| 3 | bicycle | 1 | bicycle |
| 10 | motor | 2 | motor |
| 7 | tricycle | 3 | tricycle |
| 8 | awning-tricycle | 4 | awning_tricycle |
| 4, 5, 6, 9, 0, 1, 2, 11 | car, van, truck, bus, ignored, pedestrian, people, others | — | 不在本方案处理范围 |

机动车类别（0: motor_vehicle）由其他 OBB 数据集（DroneVehicle、UAV-OBB、DOTA、VSAI）覆盖，不在本方案处理范围内。

### 类别混淆风险

| 混淆对 | 区分规则（供 VLM prompt 和人工审核使用） |
| --- | --- |
| bicycle ↔ motor | 摩托车/电动车有发动机/电池仓、排气管、后视镜、无踏板；自行车有脚踏、链条、细长车架 |
| motor ↔ tricycle | 三轮车有三个轮子、后车厢或货斗；摩托车只有两个轮子 |
| tricycle ↔ awning_tricycle | 带篷三轮车有棚顶/车厢结构，顶部封闭或半封闭；普通三轮车开放，可见驾驶员和货斗 |
| motor ↔ pedestrian（误标） | 极小的 motor HBB 可能是误标的行人；VLM 可以通过车身结构判断 |

## 7. Prompt 设计

### 设计原则

1. prompt 应针对裁剪后的局部小图（50-300px 宽高），不要用描述全景的 prompt。
2. 中英文都提供，English 用于开源模型（GroundingDINO、Florence-2），中文用于 Qwen 等中文优化模型。
3. 明确要求只分割目标主体，排除阴影、道路、行人、背景。
4. 对 tricycle 和 awning_tricycle 给出区分规则。
5. 对 motor 和 bicycle 给出区分规则。

### 7.1 bicycle 自行车

**English (GroundingDINO / Florence-2 / SAM2 text prompt):**

```
bicycle. bicycle with two wheels. bicycle with frame and pedals.
Do not include the person riding it. Do not include shadow, road, or background.
```

**English (VLM审核 / Qwen / Gemini 更详细版):**

```
Locate and segment the bicycle in this cropped image.
A bicycle has two wheels, a thin frame, handlebars, pedals, and a chain.
The bicycle may be viewed from above (drone view) so the wheels and frame are the key features.
Exclude: the rider/person, shadows on the ground, road surface, other vehicles, and background.
If the target is not a bicycle (e.g. it's a motorcycle with an engine, or a tricycle with three wheels), output "not_bicycle".
```

**中文 (Qwen / Gemini):**

```
定位并分割这张裁剪图中的自行车。
自行车有两个轮子、车架、车把、脚踏和链条，从无人机俯视角度看，轮子和车架是最明显的特征。
只分割自行车本身，不包含骑车的人、地面阴影、路面、其他车辆和背景。
如果目标不是自行车（例如是有发动机的摩托车/电动车，或者三轮车），请输出"not_bicycle"。
```

### 7.2 motor 摩托车/电动车

**English (GroundingDINO):**

```
motorcycle. motorbike. scooter. electric bike with motor.
Two-wheeled motor vehicle.
Do not include the rider. Do not include shadow, road, or background.
```

**English (VLM审核):**

```
Locate and segment the motorcycle/motorbike/scooter/electric two-wheeler in this cropped image.
A motor two-wheeler has two wheels, an engine/battery compartment, a seat, handlebars, and may have a rearview mirror and exhaust pipe. It does NOT have pedals (unlike a bicycle) and does NOT have three wheels (unlike a tricycle).
From drone view, the engine/battery housing and the absence of pedals are key distinguishing features from bicycles.
Exclude: the rider, shadows, road, background.
If the target is actually a bicycle (with pedals and chain) or a tricycle (with three wheels), output "not_motor".
```

**中文:**

```
定位并分割这张裁剪图中的摩托车/电动车/两轮机动车辆。
摩托车/电动车有两个轮子、发动机/电池仓、座椅、车把，可能有后视镜和排气管。没有脚踏板（区别于自行车），没有第三个轮子（区别于三轮车）。
从无人机俯视角度看，发动机/电池仓外壳和不含脚踏是区别于自行车的关键特征。
只分割车辆本身，不包含骑手、阴影、路面、背景。
如果目标实际上是自行车（有脚踏和链条）或三轮车（有三个轮子），请输出"not_motor"。
```

### 7.3 tricycle 三轮车

**English (GroundingDINO):**

```
tricycle. three-wheeled vehicle. three-wheeler with open cargo area.
Do not include the driver. Do not include shadow, road, or background.
```

**English (VLM审核):**

```
Locate and segment the tricycle (three-wheeled vehicle) in this cropped image.
A tricycle has THREE wheels, an open cargo area or flatbed at the rear, and the driver area is usually open/exposed. It does NOT have a closed cabin or roof/awning structure.
From drone view, the three-wheel layout and open rear cargo area are the key features.
Exclude: the driver, shadows, road, background.
If the target has a roof/awning/canopy covering the rear (awning-tricycle), output "awning_tricycle" instead of "tricycle".
If the target is a two-wheeled motorcycle or bicycle, output "not_tricycle".
```

**中文:**

```
定位并分割这张裁剪图中的三轮车（开放货斗式三轮车）。
三轮车有三个轮子，后面是开放的货斗或平板，驾驶员区域是开放/暴露的。不带棚顶/车厢。
从无人机俯视角度看，三轮布局和开放的尾部货斗是关键特征。
只分割车辆本身，不包含驾驶员、阴影、路面、背景。
如果目标有棚顶/车厢覆盖后部（带篷三轮车），请输出"awning_tricycle"而非"tricycle"。
如果目标是两轮摩托车或自行车，请输出"not_tricycle"。
```

### 7.4 awning_tricycle 带篷三轮车

**English (GroundingDINO):**

```
awning tricycle. covered three-wheeler. three-wheeled vehicle with canopy. tricycle with roof.
Do not include the driver. Do not include shadow, road, or background.
```

**English (VLM审核):**

```
Locate and segment the awning-tricycle (covered/canopied three-wheeled vehicle) in this cropped image.
An awning-tricycle has THREE wheels and a closed or semi-closed cabin/roof/awning/canopy structure covering the rear or the entire vehicle. The roof/canopy makes it look more like a small boxy vehicle from above, unlike an open tricycle.
From drone view, the covered rectangular body and three-wheel layout are the key features.
Exclude: the driver, shadows, road, background.
If the target is an open tricycle without a roof/canopy, output "tricycle" instead of "awning_tricycle".
If the target is a car/van/truck, output "not_awning_tricycle".
```

**中文:**

```
定位并分割这张裁剪图中的带篷三轮车（有棚顶/车厢的三轮车辆）。
带篷三轮车有三个轮子，后部或整体有封闭/半封闭的车厢、棚顶或遮篷结构。从俯视角度看，封闭矩形车身和三轮布局是关键特征。
只分割车辆本身，不包含驾驶员、阴影、路面、背景。
如果目标是开放的三轮车（无棚顶），请输出"tricycle"而非"awning_tricycle"。
如果目标是汽车/面包车/卡车，请输出"not_awning_tricycle"。
```

### 7.5 混淆类别排除 prompt（通用负面提示）

当 VLM 返回不确定结果时，使用以下通用排除 prompt 做二次确认：

**English:**

```
This cropped image may contain one of: bicycle, motorcycle, tricycle, awning-tricycle, car, van, truck, bus, pedestrian, or background only.
Your task:
1. Identify which (if any) vehicle is the MAIN subject centered in this crop.
2. If the main subject is bicycle/motorcycle/tricycle/awning-tricycle, segment it precisely.
3. If the main subject is a car/van/truck/bus, output "motor_vehicle" (do not segment — these are handled separately).
4. If the main subject is a pedestrian or people, output "pedestrian" (do not segment).
5. If the crop is empty or contains only road/background, output "empty".
Only segment the target vehicle body. Exclude shadows, road markings, people, and other vehicles.
```

**中文:**

```
这张裁剪图可能包含以下之一：自行车、摩托车/电动车、三轮车、带篷三轮车、小汽车、面包车、卡车、公交车、行人、或仅背景。
你的任务：
1. 识别裁剪图中心的主要目标是什么（如果有的话）。
2. 如果主要目标是自行车/摩托车/三轮车/带篷三轮车，精准分割它。
3. 如果主要目标是汽车/面包车/卡车/公交车，输出"motor_vehicle"（不需要分割——这些由其他流程处理）。
4. 如果主要目标是行人，输出"pedestrian"（不需要分割）。
5. 如果裁剪图是空的或只有路面/背景，输出"empty"。
只分割目标车辆本体。排除阴影、路面标线、行人和其他车辆。
```

## 8. Mask 转 OBB 方法

### 方法概述

与 v1 相同，使用 `cv2.minAreaRect` 从 mask 拟合旋转矩形：

```
mask (binary) → cv2.findContours / connectedComponents → 最大连通域 → cv2.minAreaRect → cv2.boxPoints → order_points_clockwise → OBB 四点
```

### 与 v1 的改进

1. **VLM/Grounding 给了 refined box**。v2 中 GroundingDINO/Florence-2 的输出 box 可以作为 SAM2 的 box prompt，比 v1 直接使用 expanded HBB 更精准。
2. **多 mask 选择**。如果 SAM2 返回多个 mask，v2 增加了 VLM 语义判断来选最佳 mask（例如选 "bicycle with full frame" 而不是 "bicycle wheel only"）。
3. **异常 mask 处理**。如果 mask 面积过小（<10px）、过大（>crop 面积的 90%）、或 mask 中心偏离 HBB 中心超过阈值，标记为低质量并进入人工审核。

### 坐标还原

```python
# 裁剪区域在全局图中的位置
ex1, ey1, ex2, ey2 = expanded_xyxy
# mask 中的局部坐标 → 全局坐标
global_x = local_x + ex1
global_y = local_y + ey1
# minAreaRect 拟合
rect = cv2.minAreaRect(global_points)
obb = cv2.boxPoints(rect)
# 顺时针排序
obb = order_points_clockwise(obb)
```

## 9. 质量评分指标

### 指标定义

v2 在 v1 的几何指标基础上增加语义指标：

#### 几何指标（继承 v1）

| 指标 | 公式 | 含义 | v1 阈值 |
| --- | --- | --- | --- |
| `area_ratio` | OBB面积 / HBB面积 | 面积比，过小漏分割，过大含背景 | 0.25 ~ 1.50 |
| `center_shift` | OBB中心到HBB中心的距离 / HBB对角线 | 中心偏移 | < 0.35 |
| `foreground_ratio` | mask前景像素 / OBB面积 | 框内前景占比 | > 0.20 |
| `aspect_ratio` | max(w,h) / min(w,h) | 长宽比 | 类别先验范围内 |
| `boundary_clip_ratio` | OBB被裁剪部分面积 / OBB面积 | 边界裁剪程度 | < 0.05 |

#### 新增语义指标（v2 新增）

| 指标 | 来源 | 含义 | 阈值 |
| --- | --- | --- | --- |
| `class_confidence` | VLM / Grounding 输出的置信度 | 类别置信度 | > 0.5 |
| `vlm_box_iou` | VLM box 与 SAM2 box 的 IoU | Grounding 和分割的一致性 | > 0.3 |
| `semantic_score` | VLM 审核输出 | VLM 对最终 OBB 的语义评分 | > 0.5 |

#### 综合评分（v2）

```python
geometry_score = 0.35 * area_score + 0.25 * center_score + 0.20 * fg_score + 0.10 * aspect_score + 0.10 * boundary_score
semantic_score = 0.60 * class_confidence + 0.40 * vlm_box_iou
final_score = 0.55 * geometry_score + 0.45 * semantic_score

if final_score >= 0.65 and geometry_score >= 0.50 and semantic_score >= 0.50:
    status = "auto_accept"
elif final_score < 0.35:
    status = "reject"
else:
    status = "review"
```

### 异常检测清单

| 异常类型 | 检测方式 | 处理 |
| --- | --- | --- |
| mask 为空 | `np.count_nonzero(mask) == 0` | reject |
| OBB 面积异常大 | `area_ratio > 3.0` | reject |
| OBB 面积异常小 | `area_ratio < 0.1` | reject |
| OBB 四点退化 | 三角形或线段，`polygon_area < 4` | reject |
| mask 全部为前景 | `foreground_ratio > 0.95`（可能分割了整个 crop） | review |
| VLM 输出与 HBB 类别冲突 | `vlm_class != hbb_class` 且 VLM 置信度 > 0.8 | review |
| 多目标粘连 | 连通域数量 > 1 且次大连通域面积 > 主连通域的 30% | review |

## 10. 审核队列设计

### 队列分层

| 层级 | 来源 | 占比估计 | 处理方式 |
| --- | --- | --- | --- |
| L0: auto_accept | 高质量样本 (final_score ≥ 0.65) | ~40-60% | 直接写入 YOLO-OBB 标签 |
| L1: random_sample | 从 auto_accept 随机抽检 2-5% | ~2-5% | 进入 Streamlit 审核网页 |
| L2: review | 中等质量样本 (0.35 ≤ final_score < 0.65) | ~25-40% | 进入 Streamlit 审核网页 |
| L3: reject_ambiguous | 低质量样本但可能可挽救 | ~5-10% | 记录，等待 VLM 二次裁决或人工修正 |
| L4: reject | 明确不可用 | ~5-10% | 记录失败原因，排除 |

### 审核网页增强（v2）

在 v1 的 `review_pseudo_obb_app.py` 基础上增加：

1. VLM 输出展示：显示 GroundingDINO 的 refined box、VLM 的类别判断和置信度。
2. 多后端对比：如果同时运行了多个后端（grounded_sam2 + florence2_sam2），可以切换查看。
3. VLM 审核结果：如果有 Qwen/Gemini 二次审核结果，展示在右侧面板。
4. 快速过滤：按 VLM 置信度、类别混淆、后端来源过滤队列。

## 11. 高质量样本入库策略

### 入库条件

样本需同时满足：

1. `final_score >= 0.65`
2. `geometry_score >= 0.50`
3. `semantic_score >= 0.50`
4. 不在异常检测清单中
5. `VLM class_confidence > 0.5` 或 `VLM 输出类别 == HBB 类别`

### 入库格式

直接写入 YOLO-OBB 标签文件（与 v1 相同）：

```
class_id x1_norm y1_norm x2_norm y2_norm x3_norm y3_norm x4_norm y4_norm
```

### 入库记录

同时写入 `quality.jsonl`，包含完整的 v2 字段（见第 13 节数据结构）。

## 12. 低质量样本人工审核策略

### 审核优先级

1. **P0（优先审核）**：VLM 类别与 HBB 类别冲突的样本（可能是 VisDrone 标注错误或 VLM 判断错误）。
2. **P1**：类内混淆样本（tricycle vs awning_tricycle, motor vs bicycle）。
3. **P2**：遮挡/截断样本（occlusion ≥ 2 或 truncation ≥ 2）。
4. **P3**：几何质量分偏低但 VLM 语义分高的样本。
5. **P4**：随机抽检样本。

### 审核工具

继续使用 v1 的 Streamlit 审核网页（`review_pseudo_obb_app.py`），v2 需新增以下展示字段：

- VLM 后端名称
- VLM 类别判断和置信度
- VLM box 与 SAM2 mask 的一致性（IoU）
- 多后端对比视图（如有）

### 审核后处理

使用 `apply_pseudo_obb_review.py`（与 v1 相同），根据审核决定重新生成 YOLO-OBB 标签。

## 13. 与现有 v1 方案的区别

| 维度 | v1 | v2 |
| --- | --- | --- |
| 分割模型 | SAM / SAM2 / GrabCut | GroundingDINO + SAM2 / Florence-2 + SAM2 |
| 分割 prompt | HBB expanded box 作为唯一 prompt | 文本 prompt + HBB expanded box 双 prompt |
| 语义理解 | 无（依赖 VisDrone 原始 class_id） | 有（VLM 可判断类别、区分混淆类） |
| 类别确认 | 继承 VisDrone HBB label | VLM 语义审核 + HBB label 交叉验证 |
| 质量评分 | 纯几何指标 | 几何 + 语义双维度评分 |
| 审核队列 | 按质量分二元分类 | 按质量分 + VLM 置信度分层 |
| 裁剪外扩比例 | 10% | 20%（给 VLM 更多上下文） |
| 输出目录 | `visdrone_pseudo_obb_v1` / `visdrone_yolo_obb_v1` | `visdrone_pseudo_obb_v2` / `visdrone_yolo_obb_v2` |
| 中间格式 | quality.jsonl (v1 schema) | quality.jsonl (v2 expanded schema) |
| 代码文件 | `visdrone_hbb_to_pseudo_obb.py` | 新增 `vlm_pseudo_obb_v2.py`，复用解析/选择/拟合/评分/审核框架 |

## 14. Codex 后续需要执行的视觉测试任务

以下任务需要真实图像视觉判断，不是纯代码审查可以完成的。请在 v2 实现后由 Codex 或人工执行：

### 14.1 分割质量对比

- [ ] 从 val 中随机抽取每类 50 个样本，同时运行 v1 (SAM2 box-prompt) 和 v2 (Grounded-SAM-2 text+box prompt)。
- [ ] 并排展示 v1 和 v2 的 mask + OBB 可视化结果。
- [ ] 人工判断：v2 的 mask 是否比 v1 更贴合目标？特别是在以下困难场景：
  - 目标 < 30px 的小目标
  - 多目标靠近/粘连
  - 部分遮挡
  - 阴影覆盖
  - tricycle vs awning_tricycle 区分

### 14.2 VLM 类别判断准确率

- [ ] 从 val 中抽取每类 100 个样本，运行 VLM（GroundingDINO / Qwen / Gemini）做类别判断。
- [ ] 与 VisDrone 原始 HBB 标签对比，计算一致率。
- [ ] 对不一致样本，人工判断哪个更正确。
- [ ] 报告哪些类别对最容易混淆（如 motor vs bicycle, tricycle vs awning_tricycle）。

### 14.3 OBB 贴合度

- [ ] 从 auto_accept 样本中抽取每类 30 个，人工判断 OBB 四点是否贴合目标。
- [ ] 从 review 样本中抽取每类 30 个，人工判断是否可以接受。
- [ ] 报告 OBB 贴合度合格率（与人工标注对比）。

### 14.4 Prompt 有效性

- [ ] 测试不同 prompt 模板在 GroundingDINO 上的检测率。
- [ ] 对比简单 prompt（如 "bicycle"）和详细 prompt（含区分规则）的效果差异。
- [ ] 确定最佳 prompt 模板。

### 14.5 后端对比

- [ ] GroundingDINO vs Florence-2 在 VisDrone 小目标上的检测率对比。
- [ ] 有 Grounding vs 无 Grounding（纯 SAM2 box-prompt）的 mask 质量对比。
- [ ] YOLO-World 做初筛的可行性评估。

## 15. 风险点和待确认问题

### 已识别风险

| 风险 | 影响 | 缓解措施 |
| --- | --- | --- |
| GroundingDINO 在 <50px 目标上检测率低 | 大量样本退化为纯 SAM2 box-prompt | 设置最小 crop 尺寸阈值；太小目标直接使用 HBB expanded box 作为 SAM2 prompt |
| VLM 推理成本高（GPU 显存、API 费用） | 全量 8629 个目标（仅 val）耗时长 | 先小批量实验验证性价比；考虑 YOLO-World 做高速初筛 |
| VLM 类别输出与 VisDrone 标签不一致 | 导致类别标注混乱 | 不一致样本统一进入人工审核，不自动覆盖 |
| SAM2 在密集小目标上 mask 粘连 | 一个 mask 覆盖多个目标 | 用 GroundingDINO 的 refined box 缩小 SAM2 prompt 范围 |
| tricycle/awning_tricycle 从俯视角度难以区分 | 类别准确率低 | VLM 语义判断 + 人工审核双保险；准备好 4 类退化版本 |
| 多后端版本管理和依赖冲突 | 环境配置复杂 | 每个后端独立环境或 Docker 容器；统一 JSONL 中间格式解耦 |

### 待确认问题

1. GroundingDINO 权重选择：Swin-T（轻量）还是 Swin-B（精度）？建议先用 Swin-T 测试速度，再决定是否需要 Swin-B。
2. SAM2 模型规模：tiny/small/base_plus/large？参考 v1 经验，base_plus 可能在精度和速度间平衡较好。
3. VLM API 调用：如果是 Qwen/Gemini 在线 API，需要确认预算、速率限制、数据隐私合规。
4. 是否需要引入 Grounded-SAM-2 的官方整合仓库，还是分别部署 GroundingDINO 和 SAM2。
5. 多后端并行时，每个样本保存几个后端的结果？建议所有后端结果都保存在 JSONL 中，审核时可以选择最佳后端。
6. 是否需要为 v2 创建独立的 conda 环境（如 `openvter-obb-vlm`），避免与 v1 的 `openvter-obb-sam` 冲突。
7. VisDrone test-dev（1610 张，无公开标注）是否作为无监督伪标注数据源？建议暂不使用，除非实验证明 v2 在 val 上的质量足够高。
