# 公开数据源梳理：机动车 OBB 与弱势交通参与者 HBB

## 1. 可直接支撑机动车 OBB 的数据集

| 数据集 | 框类型 | 已知类别/特点 | 对本项目的用途 | 局限 |
| --- | --- | --- | --- | --- |
| DroneVehicle | OBB | `car`, `truck`, `bus`, `van`, `freight car`；56,878 张 RGB/红外图像，五类车辆 OBB | 作为 `motor_vehicle` 粗类或机动车 OBB 预训练数据 | 不覆盖自行车、摩托车、三轮车等弱势交通参与者 |
| UAV-OBB | OBB | 最新 Mendeley v4：1,617 张 1920x1080 RGB 图像，46,807 个实例，`bike`, `bus`, `car`, `other_vehicle`, `taxi`, `truck` | 可补充机动车和部分两轮车/粗粒度车辆 OBB | `bike` 与 `other_vehicle` 对本项目类别过粗，不能直接覆盖三轮车/带篷三轮车 |
| DOTA | OBB / 四点多边形 | 遥感/航拍大规模 OBB；交通相关类别主要是 `small vehicle`, `large vehicle` | 作为旋转框检测的通用预训练或 OBB 格式参考 | 分辨率、拍摄高度和类别粒度与城市 UAV 交通场景差异较大 |
| VSAI | OBB / 任意四边形 | 小型车辆、大型车辆，含遮挡率标注 | 适合增强机动车旋转框定位能力 | 类别粗，不覆盖弱势交通参与者 |
| EAGLE | HBB / RBB / OBB | 航空影像车辆方向检测，包含小型/大型车辆 | 可作为车辆有向框预训练参考 | 航空遥感域偏强，类别不满足本项目弱势交通参与者需求 |

## 2. 支撑弱势交通参与者类别的 HBB 数据集

| 数据集 | 框类型 | 已知类别/特点 | 对本项目的用途 | 局限 |
| --- | --- | --- | --- | --- |
| VisDrone-DET | HBB | `pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor` | 最重要的弱势交通参与者类别来源，可生成伪 OBB | 原始标注没有方向角和四点框 |
| 自有 OpenVTER 视频 | 当前依赖已有模型输出 | 更贴近本地域、本项目相机高度和道路结构 | 用于补充困难样本、做人工复核和领域适配 | 需要额外标注或伪标注质量筛选 |

## 3. 建议的数据组合

不要追求单一公开数据集覆盖全部类别。推荐组合为：

1. 用 DroneVehicle / UAV-OBB / DOTA / VSAI / EAGLE 学习机动车 OBB 定位能力。
2. 用 VisDrone 的 HBB 标签保留弱势交通参与者类别，生成 `bicycle`, `motor`, `tricycle`, `awning_tricycle` 的伪 OBB。
3. 用自有 OpenVTER 视频补充中国本地域路口、环岛、非机动车混行等困难样本。
4. 最终训练 5 类 OBB：`motor_vehicle`, `bicycle`, `motor`, `tricycle`, `awning_tricycle`。

## 4. 参考来源

- DroneVehicle: https://github.com/VisDrone/DroneVehicle
- UAV-OBB: https://data.mendeley.com/datasets/6snrjwcpkh
- DOTA: https://captain-whu.github.io/DOTA/dataset.html
- VSAI: https://datasetninja.com/vsai
- EAGLE: https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/eagle
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
