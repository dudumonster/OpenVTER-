# VisDrone HBB 转伪 OBB 协议

VisDrone 的价值在于类别覆盖：它包含自行车、摩托车/电动车、三轮车、带篷三轮车等弱势交通参与者。问题是它的检测标注是 HBB，不能直接训练 OBB 检测器。因此建议把 VisDrone 用作“类别标签来源”，再通过弱监督方式生成伪 OBB。

## 1. 最简可落地流程

```text
VisDrone HBB 标注
  -> 按类别裁剪目标区域
  -> 前景分割或边缘提取
  -> minAreaRect 拟合旋转矩形
  -> 类别和几何规则过滤
  -> 人工抽检/修正低质量样本
  -> 导出 DOTA/YOLO-OBB/MMRotate 可训练格式
```

## 2. 推荐转换方法

### 方法 A：HBB 裁剪 + 前景分割 + 最小外接旋转矩形

这是最直接的做法。

1. 根据 VisDrone 的 HBB 裁剪目标，向外扩 5%-15% 边界，避免车身边缘被截断。
2. 在裁剪图内使用 SAM、GrabCut、轻量语义分割模型或传统边缘方法提取目标前景。
3. 对最大连通域或目标主体轮廓使用 `minAreaRect` 拟合旋转矩形。
4. 把局部坐标还原到原图坐标，得到四点 OBB。
5. 通过面积比例、长宽比、中心偏移、前景覆盖率过滤明显错误样本。

优点：实现简单，能快速得到第一版伪标签。

缺点：小目标、阴影、遮挡、多人/多车粘连时容易失败。

### 方法 B：已有模型辅助的教师伪标注

你已经有一个机动车检测效果不错的模型，可以把它作为教师模型的一部分。

1. 对机动车类，优先使用已有 OBB/车辆检测模型输出，统一映射为 `motor_vehicle`。
2. 对 VisDrone 中的弱势交通参与者，使用 HBB 标签定位候选区域。
3. 在候选区域内做分割和 OBB 拟合。
4. 如果已有模型也能给出车辆方向，可以用它修正机动车粗类的 OBB 长轴方向。
5. 低置信度或几何异常样本进入人工复核队列。

优点：能充分利用现有成果，不必把机动车部分重复做一遍。

缺点：对自行车、三轮车、带篷三轮车仍然需要伪 OBB 或人工修正。

### 方法 C：视频轨迹方向修正

如果数据来自视频或能形成连续帧，可以用运动方向修正 OBB 朝向。

1. 对同一目标建立短轨迹。
2. 计算连续中心点的主运动方向。
3. 当目标速度高于阈值时，用运动方向约束 OBB 长轴。
4. 对静止或低速目标，不强行使用运动方向，只保留几何拟合结果。

优点：能减少 90 度翻转和方向不一致问题。

缺点：只适用于视频数据；静止目标帮助有限。

## 3. 三轮车与带篷三轮车是否能区分

可以尝试区分，而且建议先保留这两个类别。

原因是 VisDrone 本身已经区分 `tricycle` 和 `awning-tricycle`，转换 OBB 时类别标签可以直接继承。模型学习时，带篷三轮车通常具有更明显的车厢/棚顶轮廓，外形更接近小矩形车辆；普通三轮车更开放，轮廓更窄、更不规则。

但这里有两个风险：

1. 小目标条件下，棚顶细节可能只有几个像素，类别容易混淆。
2. 不同地区的电动三轮车形态差异很大，VisDrone 类别定义不一定完全等同于本地数据。

因此建议实验时保留两个版本：

| 版本 | 类别设计 | 用途 |
| --- | --- | --- |
| 5 类版本 | `motor_vehicle`, `bicycle`, `motor`, `tricycle`, `awning_tricycle` | 主实验，验证能否区分带篷/不带篷三轮车 |
| 4 类退化版本 | `motor_vehicle`, `bicycle`, `motor`, `tricycle_all` | 消融实验，如果带篷三轮车 AP 不稳定则作为稳健方案 |

## 4. 质量筛选规则

建议每个伪 OBB 样本保存质量分数，低分样本不直接进入训练集。

| 指标 | 筛选含义 |
| --- | --- |
| `area_ratio` | OBB 面积 / HBB 面积，过小可能漏分割，过大可能包含背景 |
| `center_shift` | OBB 中心相对 HBB 中心偏移，过大说明目标提取偏了 |
| `foreground_ratio` | 前景像素 / OBB 面积，过低说明 OBB 包含太多背景 |
| `aspect_ratio` | 长宽比是否符合类别先验 |
| `track_consistency` | 视频中相邻帧方向是否连续 |

建议第一版阈值不要过严，先保留高质量样本训练一个初版模型，再用模型反推和人工复核迭代。

## 5. 最终导出格式

建议内部统一保存八参数四点格式：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4 quality source_image source_dataset
```

之后再按需要导出：

- DOTA 格式：适配 MMRotate / DOTA 评测；
- YOLO-OBB 格式：适配 Ultralytics OBB；
- OpenVTER 推理格式：适配项目内部四点框接口。

## 6. 当前 v1 实现

当前已经实现一版可运行流水线：

```bash
python3 dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py validate
python3 dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py generate --splits train val --segmenter auto
streamlit run dataset_construction/scripts/review_pseudo_obb_app.py
python3 dataset_construction/scripts/apply_pseudo_obb_review.py
```

### 类别编号

导出 YOLO-OBB 时使用全局 5 类编号，而不是重新从 0 开始编号。这样后续可以直接和机动车 OBB 数据组合：

| YOLO-OBB 类别 ID | 类别 | 来源 |
| --- | --- | --- |
| 0 | `motor_vehicle` | DroneVehicle / DOTA / VSAI / UAV-OBB |
| 1 | `bicycle` | VisDrone `class_id=3` |
| 2 | `motor` | VisDrone `class_id=10` |
| 3 | `tricycle` | VisDrone `class_id=7` |
| 4 | `awning_tricycle` | VisDrone `class_id=8` |

### 默认质量规则

v1 使用以下规则判断是否进入审核队列：

- `area_ratio = OBB 面积 / HBB 面积` 小于 0.25 或大于 1.50；
- `center_shift = OBB 中心相对 HBB 中心偏移 / HBB 对角线` 大于 0.35；
- `foreground_ratio = mask 前景像素 / OBB 面积` 小于 0.20；
- 长宽比超出类别先验；
- mask 为空、OBB 被边界裁剪、严重遮挡或截断。

高质量样本直接写入 YOLO-OBB 标签；低质量样本写入 `review_queue.jsonl`。审核网页的决策会保存到 `review_decisions.jsonl`，随后用 `apply_pseudo_obb_review.py` 重新生成标签。

### SAM/SAM2 与 fallback

如果服务器已经安装 Segment Anything 并准备好权重，可以指定：

```bash
python3 dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py generate \
  --segmenter sam \
  --sam-checkpoint /path/to/sam_vit_h.pth \
  --sam-model-type vit_h
```

如果没有 SAM/SAM2 权重，脚本会使用 OpenCV GrabCut 作为 fallback。这一模式适合先跑通数据流、审核网页和 YOLO-OBB 标签格式；正式实验建议在服务器 GPU 上使用 SAM/SAM2 重新生成一版伪标签。
