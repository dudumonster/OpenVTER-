# OpenVTER OBB 数据集构建工作区

本目录用于沉淀“面向无人机交通目标的 OBB 数据集构建”方案。它不直接存放大型原始数据集，而是存放数据源清单、类别映射、标注转换协议和后续实验记录。

## 当前类别策略

现阶段不建议直接训练 9 类 OBB。更稳妥的路线是先训练 5 类 OBB：

| OBB 训练类别 | 中文含义 | 后续处理 |
| --- | --- | --- |
| `motor_vehicle` | 机动车粗类 | 结合标定后的物理车长，细分为小汽车、面包车、卡车、公交车、大型货车 |
| `bicycle` | 自行车 | 直接作为最终类别 |
| `motor` | 摩托车 / 电动车 | 直接作为最终类别，必要时在论文中说明电动车与摩托车合并 |
| `tricycle` | 三轮车 | 直接作为最终类别 |
| `awning_tricycle` | 带篷三轮车 | 直接作为最终类别 |

这个设计的核心是：机动车类别内部外观差异可以由几何规则辅助细分，而弱势交通参与者之间的形态差异必须尽量由检测模型直接学习。

## 推荐目录结构

```text
dataset_construction/
  README.md
  metadata/
    obb_vehicle_sources.md        # 公开 OBB / HBB 数据源梳理
  protocols/
    visdrone_hbb_to_obb.md        # VisDrone 水平框转伪 OBB 的流程
  schemas/
    class_mapping.yaml            # 训练类别与 OpenVTER 输出类别映射
```

后续如果真正下载或生成数据，建议使用以下外部结构，并把大文件加入 `.gitignore`：

```text
data/
  raw/                            # 原始公开数据集，不进 Git
  interim/                        # 中间转换结果，不进 Git
  obb_traffic_5cls/
    images/
    labels/
    splits/
    manifests/
```

## 当前工作判断

1. 已有机动车 OBB / 车辆检测模型可以继续保留，不需要推倒重来。
2. 公开 OBB 数据主要支撑机动车或粗粒度车辆，适合作为 `motor_vehicle` 的定位预训练来源。
3. VisDrone 覆盖自行车、摩托车/电动车、三轮车、带篷三轮车等弱势交通参与者，但标注是水平框，因此更适合作为“类别来源 + 伪 OBB 生成来源”。
4. 三轮车和带篷三轮车能不能分开，主要取决于样本质量和标注一致性。VisDrone 已经给出二者类别标签，转换 OBB 时应保留这个标签；真正困难的是伪 OBB 几何是否贴合目标，而不是类别名本身。

## VisDrone HBB 转伪 OBB v1

第一版实现位于：

```text
dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py
dataset_construction/scripts/review_pseudo_obb_app.py
dataset_construction/scripts/apply_pseudo_obb_review.py
```

先检查 VisDrone 目录：

```bash
python3 dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py validate
```

生成伪 OBB。没有 SAM/SAM2 权重时默认使用 OpenCV GrabCut 跑通流程；如果有 SAM 权重，可以通过 `--segmenter sam --sam-checkpoint /path/to/sam.pth` 使用 box-prompt 分割。

```bash
python3 dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py generate \
  --splits train val \
  --segmenter auto \
  --copy-mode symlink
```

输出目录：

```text
dataset_construction/derived/visdrone_pseudo_obb_v1/
  quality.jsonl          # 全量伪标签质量记录
  review_queue.jsonl     # 低质样本 + 随机抽样审核队列
  masks/                 # mask crop
  previews/              # 审核预览图

dataset_construction/derived/visdrone_yolo_obb_v1/
  data.yaml
  images/train|val/      # 默认软链接到 raw 图片
  labels/train|val/      # YOLO-OBB 标签
```

启动审核网页：

```bash
streamlit run dataset_construction/scripts/review_pseudo_obb_app.py
```

审核后重新应用结果：

```bash
python3 dataset_construction/scripts/apply_pseudo_obb_review.py
```

### 环境配置记录

本机推荐使用独立环境 `openvter-obb-sam`，不要在原有 `dms-yolo` 中继续混装 SAM/SAM2 依赖：

```bash
conda create -n openvter-obb-sam python=3.10 -y
conda activate openvter-obb-sam
pip install torch torchvision opencv-python pyyaml pandas streamlit segment-anything watchdog ultralytics
```

该环境已经验证可用于：

- VisDrone 数据读取检查；
- SAM ViT-B + MPS 伪 OBB 生成；
- OpenCV GrabCut fallback 伪 OBB 生成；
- YOLO-OBB 标签校验；
- Streamlit 审核网页。

本机 SAM v1 权重已放在：

```text
checkpoints/sam/sam_vit_b_01ec64.pth
```

已验证 Mac 本机 MPS 可以运行 SAM ViT-B，小样本测试 5 张 VisDrone val 图耗时约 9 秒。正式全量运行可以先在本机跑 `val`，确认效果后再跑 `train`。

本机启动审核网页：

```bash
dataset_construction/scripts/run_review_app.sh
```

本机运行 SAM 伪标注：

```bash
dataset_construction/scripts/run_visdrone_sam_pseudo_obb.sh
```

如果要先小批量测试：

```bash
dataset_construction/scripts/run_visdrone_sam_pseudo_obb.sh --splits val --max-images 20
python3 dataset_construction/scripts/validate_yolo_obb_dataset.py dataset_construction/derived/visdrone_yolo_obb_v1
```

服务器 CUDA 环境建议单独创建，避免影响 OpenVTER 原有环境：

```bash
conda create -n obb-sam python=3.10 -y
conda activate obb-sam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pyyaml pandas streamlit ultralytics segment-anything watchdog
```

SAM v1 服务器运行示例：

```bash
python3 dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py generate \
  --splits train val \
  --segmenter sam \
  --sam-checkpoint checkpoints/sam/sam_vit_b_01ec64.pth \
  --sam-model-type vit_b \
  --device cuda \
  --copy-mode symlink
```

如果服务器需要 SAM2，建议在独立环境中安装，避免它升级 Torch 后影响现有 YOLO/OpenVTER 环境：

```bash
pip install sam2
```
