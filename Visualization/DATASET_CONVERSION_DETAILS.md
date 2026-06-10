# 数据集转换结果与轨迹清洗逻辑说明

本文档说明当前 `Visualization/app/converter.py` 的真实转换逻辑，重点包括输入数据、输出目录、CSV 字段、质量报告、过滤规则、补全修复和平滑计算。

## 一、输入数据来源

当前转换模块只把 `det_bbox_result_*.pkl` 作为正式轨迹数据源。`raw_det` 不作为最终轨迹主数据源，`*_stab.pkl` 只记录为稳像文件，不参与轨迹动力学计算。

`det_bbox_result_*.pkl` 顶层通常包括：

```text
video_info
output_info
traj_info
process_time
raw_det
```

转换主数据来自 `traj_info`。每一帧结构为：

```python
(frame_id, output_frame, array)
```

或：

```python
(frame_id, output_frame, array, frame_time)
```

其中 `array` 是当前帧所有目标的矩阵。当前项目按前 20 列解析：

```text
0  q1_x
1  q1_y
2  q2_x
3  q2_y
4  q3_x
5  q3_y
6  q4_x
7  q4_y
8  confidence
9  category_id
10 object_id
11 world_q1_x
12 world_q1_y
13 world_q2_x
14 world_q2_y
15 world_q3_x
16 world_q3_y
17 world_q4_x
18 world_q4_y
19 lane_id
```

像素四点框 `q1_x ~ q4_y` 主要用于估算 `orthoPxToMeter` 以及后端可视化时的 world/pixel 映射。轨迹中心、尺寸、速度、加速度主要基于世界坐标四点框 `world_q1_x ~ world_q4_y` 计算。

## 二、输出目录与版本

转换后会生成中间完整结果和正式精简结果。

### Adjusted results

每个原始地点文件夹会输出两个完整版本：

```text
Visualization/Adjusted results/<folderName>/
├── full/
└── moving_filtered/
```

根目录 `Visualization/Adjusted results/<folderName>/` 不直接保存 CSV、日志或报告文件，只保留 `full` 和 `moving_filtered` 两个子文件夹。

`full` 是完整版本，保留所有通过基础清洗、补全、修复和平滑后的最终轨迹。

`moving_filtered` 是过滤版本。它基于 `full` 结果先删除高置信疑似 ID 断裂组，再按标准车辆、轻型弱势车辆、行人与自行车三类静止门控删除长期静止轨迹。该过滤不会修改 `full`。

### Final Data

当前转换还会从 `moving_filtered` 导出正式数据：

```text
Visualization/Final Data/<folderName>/
├── <folderName>_recordingMeta.csv
├── <folderName>_tracksMeta.csv
└── <folderName>_tracks.csv
```

正式导出的来源由代码中的常量控制：

```python
FINAL_DATA_SOURCE_VERSION = "moving_filtered"
```

`Final Data` 是当前可视化后端默认扫描和读取的数据目录。它只保留标准三张 CSV，不包含 `id_mapping.csv`、`filter_report.csv`、`metadata.json`、`conversion_log.txt` 或 `quality_report.json`。这些追溯和质量文件保留在 `Adjusted results` 中。

## 三、运行入口与日志

转换全部数据集：

```powershell
python "Visualization\app\converter.py" --force
```

转换指定数据集：

```powershell
python "Visualization\app\converter.py" --datasets cao_qiao_001 --force
python "Visualization\app\converter.py" --datasets cao_qiao_001 qian_qi_neng_yuan_020 --force
```

指定输入、中间输出和正式输出目录：

```powershell
python "Visualization\app\converter.py" --source-root "Visualization\Initial results" --output-root "Visualization\Adjusted results" --final-output-root "Visualization\Final Data" --force
```

检查 pkl 结构：

```powershell
python "Visualization\app\converter.py" --inspect "Visualization\Initial results\cao_qiao_001\det_bbox_result_cao_qiao_001.pkl"
```

总转换日志写入：

```text
Visualization/logs/dataset_conversion.log
```

每个完整版本自己的日志写入：

```text
Visualization/Adjusted results/<folderName>/<version>/conversion_log.txt
```

可视化后端启动：

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

可视化服务日志写入：

```text
Visualization/logs/visualization_server.log
```

## 四、每个 Adjusted results 版本的输出文件

`full` 和 `moving_filtered` 目录下都会输出：

```text
<folderName>_recordingMeta.csv
<folderName>_tracksMeta.csv
<folderName>_tracks.csv
id_mapping.csv
filter_report.csv
metadata.json
conversion_log.txt
quality_report.json
```

`id_mapping.csv` 记录每个原始 `raw_object_id` 到当前版本最终 `trackId/object_id` 的映射；被过滤的 raw tracklet 也会保留记录并写明原因。

`filter_report.csv` 记录未进入当前版本的 raw tracklet 及过滤原因。

`metadata.json` 记录版本、过滤开关、过滤参数和统计文件名。

`quality_report.json` 记录转换质量、缺帧、补帧、类别跳变、静止过滤、疑似 ID 断裂过滤等辅助统计。

## 五、recordingMeta.csv 字段

`Adjusted results` 与 `Final Data` 的 `recordingMeta.csv` 字段一致：

```text
recordingId,locationId,frameRate,numFrames,duration,numTracks,numVehicles,numVRUs,classTrackCounts,orthoPxToMeter
```

| 字段 | 计算逻辑 |
| --- | --- |
| `recordingId` | 从文件夹名解析。若 `cao_qiao_001`，最后一段 `001` 是编号，则 `recordingId = 001`。 |
| `locationId` | 从文件夹名解析。若 `cao_qiao_001`，则 `locationId = cao_qiao`。如果文件夹名不符合最后一段为数字的规则，则二者都使用文件夹名，并写 warning。 |
| `frameRate` | 优先读取 `output_info["output_fps"]`，如果没有则使用 `29.97`。 |
| `numFrames` | 优先读取 `video_info[0]["total_frames"]`。如果没有，则根据 `traj_info` 帧号推断。 |
| `duration` | `numFrames / frameRate`，单位秒。 |
| `numTracks` | 当前版本最终轨迹数量，按唯一最终 `trackId` 计数。 |
| `numVehicles` | 按最终 `class` 统计机动车轨迹数量：`car, truck, bus, freight_car, van, motor, tricycle, awning-tricycle`。 |
| `numVRUs` | 按最终 `class` 统计弱势交通参与者轨迹数量：`pedestrian, people, bicycle, tricycle, awning-tricycle, motor`。`motor/tricycle/awning-tricycle` 会同时计入车辆和 VRU。 |
| `classTrackCounts` | JSON 字符串，记录所有类别的最终轨迹数量。所有类别都会出现，即使数量为 0。 |
| `orthoPxToMeter` | 基于像素四点框边长和世界四点框边长的比例估算。对所有有效边计算 `world_edge_length / pixel_edge_length`，过滤 IQR 异常后取中位数。有效边少于 30 时输出空值并写 warning。 |

## 六、tracksMeta.csv 字段

### Adjusted results

完整版本字段：

```text
recordingId,trackId,raw_object_id,initialFrame,finalFrame,numFrames,startXCenter,startYCenter,endXCenter,endYCenter,startLaneId,endLaneId,width,length,raw_mean_width,raw_mean_height,corrected_width,corrected_height,box_orientation_source,missing_ratio,class
```

### Final Data

正式精简字段：

```text
recordingId,trackId,initialFrame,finalFrame,numFrames,startXCenter,startYCenter,endXCenter,endYCenter,startLaneId,endLaneId,width,length,class
```

字段说明：

| 字段 | 计算逻辑 |
| --- | --- |
| `recordingId` | 同 `recordingMeta.csv`。 |
| `trackId` | 当前版本内重新编号，从 1 开始连续编号。排序依据为轨迹第一帧和原始 `object_id`。 |
| `raw_object_id` | 仅 `Adjusted results` 保留。pkl 原始结果中的 `object_id`，用于追溯最终 ID 来源。 |
| `initialFrame` | 当前最终轨迹输出记录中的最小 `frame`。如果有补全帧，补全帧也参与最小值计算。 |
| `finalFrame` | 当前最终轨迹输出记录中的最大 `frame`。 |
| `numFrames` | 当前最终轨迹实际输出到 `tracks.csv` 的记录数，包括插值补全帧。 |
| `startXCenter/startYCenter` | 该轨迹第一条输出记录的平滑后中心点。 |
| `endXCenter/endYCenter` | 该轨迹最后一条输出记录的平滑后中心点。 |
| `startLaneId/endLaneId` | 第一条和最后一条输出记录的 `lane_id`；无法判断时为 `-1`。 |
| `width/length` | 最终导出和可视化使用的轨迹级稳定尺寸，单位 m。除 `pedestrian` 外，`length` 沿 `heading` 方向。 |
| `raw_mean_width/raw_mean_height` | 仅 `Adjusted results` 保留。补帧前真实检测帧中有效尺寸的轨迹级统计值。 |
| `corrected_width/corrected_height` | 仅 `Adjusted results` 保留。无向框旋转修正后的尺寸，当前与 `width/length` 同步。 |
| `box_orientation_source` | 仅 `Adjusted results` 保留。当前无向水平框修正为 `heading_corrected_from_hbb`。 |
| `missing_ratio` | 仅 `Adjusted results` 保留。该原始 `object_id` 基于补帧前唯一帧号统计的整体缺帧比例。 |
| `class` | 轨迹最终类别。按该最终轨迹真实检测帧的 `raw_class` 众数确定，不统计插值帧。 |

类别众数规则：

1. 只统计非插值帧。
2. 出现次数最多的类别作为最终 `class`。
3. 如果次数相同，则比较这些类别的 `confidence` 总和，较高者胜出。
4. 如果次数和置信度总和仍相同，则取最早出现的类别，并写 warning。
5. 如果最终类别占比 `< 0.7`，记录为类别不稳定轨迹，但不删除。

## 七、tracks.csv 字段

### Adjusted results

完整版本字段：

```text
recordingId,trackId,raw_object_id,lane_id,frame,trackLifetime,xCenter,yCenter,heading,width,length,raw_mean_width,raw_mean_height,corrected_width,corrected_height,box_orientation_source,is_interpolated,missing_ratio,xVelocity,yVelocity,xAcceleration,yAcceleration,lonVelocity,latVelocity,lonAcceleration,latAcceleration
```

### Final Data

正式精简字段：

```text
recordingId,trackId,lane_id,frame,trackLifetime,xCenter,yCenter,heading,width,length,xVelocity,yVelocity,xAcceleration,yAcceleration,lonVelocity,latVelocity,lonAcceleration,latAcceleration
```

字段说明：

| 字段 | 计算逻辑 |
| --- | --- |
| `recordingId` | 同 `recordingMeta.csv`。 |
| `trackId` | 与 `tracksMeta.csv` 中的最终 `trackId` 对应。 |
| `raw_object_id` | 仅 `Adjusted results` 保留。当前最终轨迹对应的原始 pkl `object_id`。一个 raw ID 只会对应一个最终 ID，不会把多个 raw ID 合并成一个最终 ID。 |
| `lane_id` | 原始 `traj_info` 第 19 列。补全帧会根据前后帧 lane 填充。无法判断时为 `-1`。导出 `Final Data` 时会对局部 `-1` 做邻域修正。 |
| `frame` | 原始帧号，不重新从 0 或 1 编号。补全帧使用对应缺失帧号。 |
| `trackLifetime` | 当前最终轨迹内第几条输出记录，从 1 开始递增。按实际输出记录数递增，不用 `frame - initialFrame + 1`。 |
| `xCenter/yCenter` | 世界四点框中心点经过异常修复、插值补全、Savitzky-Golay 平滑后的坐标，单位 m。 |
| `heading` | 基于平滑后的中心点计算，单位 deg。定义为 `0° -> +Y, 90° -> +X, 180° -> -Y, 270° -> -X`。 |
| `width/length` | 该 `trackId` 的修正后稳定宽度和长度，所有帧相同，单位 m。 |
| `raw_mean_width/raw_mean_height` | 仅 `Adjusted results` 保留。补帧前真实检测帧的轨迹级尺寸统计。 |
| `corrected_width/corrected_height` | 仅 `Adjusted results` 保留。可视化和导出旋转框使用的修正后尺寸，与 `width/length` 同步。 |
| `box_orientation_source` | 仅 `Adjusted results` 保留。当前无向框修正来源为 `heading_corrected_from_hbb`。 |
| `is_interpolated` | 仅 `Adjusted results` 保留。当前记录是否由缺失帧插值生成。 |
| `missing_ratio` | 仅 `Adjusted results` 保留。该轨迹补帧前的整体缺帧比例。 |
| `xVelocity/yVelocity` | 基于平滑后中心点对真实时间差分得到，单位 m/s。 |
| `xAcceleration/yAcceleration` | 基于速度对真实时间差分得到，单位 m/s²。 |
| `lonVelocity/latVelocity` | 沿 heading 前进方向和垂直 heading 方向的速度，单位 m/s。`latVelocity` 正值表示目标向自身左侧运动。 |
| `lonAcceleration/latAcceleration` | 沿 heading 前进方向和垂直 heading 方向的加速度，单位 m/s²。 |

当前代码不再输出旧文档中提到的 `centerX/centerY` 兼容字段。

## 八、id_mapping.csv 与 filter_report.csv

`id_mapping.csv` 每个 `Adjusted results` 版本都会输出，字段顺序固定：

```text
dataset_id,version,raw_object_id,final_object_id,class_name_mode,start_frame,end_frame,total_frames,mean_confidence,is_kept,is_filtered,filter_type,filter_reason,fragmentation_group_id,quality_score
```

`raw_object_id` 是原始 pkl / `traj_info` 第 10 列中的 `object_id`。`final_object_id` 是当前版本重新分配后的最终 ID，也就是前端默认显示、筛选和跳转使用的 `object_id`，标准 CSV 中对应 `trackId`。被过滤的 raw tracklet 会在 `id_mapping.csv` 中保留记录，`final_object_id` 为空，并写明 `filter_type/filter_reason`。

`filter_report.csv` 记录当前版本未保留的 raw tracklet，字段顺序固定：

```text
dataset_id,version,raw_object_id,filter_type,filter_reason,fragmentation_group_id,related_raw_object_ids,fragmentation_score,quality_score,start_frame,end_frame,total_frames,class_name_mode
```

疑似 ID 断裂组的统一删除原因是：

```text
suspected_id_fragmentation_drop_all_related_tracklets
```

该策略不会做 tracklet stitching，不会把两个 `raw_object_id` 合并成同一个最终 `object_id`，也不会保留 `quality_score` 最高或 `total_frames` 最长的片段。

## 九、过滤参数

### 静止过滤

静止过滤参数位于 `converter.py` 的三组配置：`STATIC_GATE`、`LIGHT_VRU_STATIC_GATE` 和 `VRU_STATIC_GATE`。

```python
STATIC_VEHICLE_CLASSES = {"car", "truck", "bus", "freight_car", "van"}

STATIC_GATE = {
    "min_track_length": 30,
    "max_displacement": 1.0,
    "max_path_length": 2.0,
    "max_stationary_extent": 2.0,
    "max_mean_speed": 0.2,
    "static_ratio_threshold": 0.8,
    "per_frame_motion_threshold": 0.05,
    "filter_classes": sorted(STATIC_VEHICLE_CLASSES),
}

LIGHT_VRU_STATIC_GATE = {
    "motor": {
        "min_track_length": 60,
        "max_stationary_extent": 1.5,
        "required_signals": ["low_stationary_extent", "low_mean_speed", "high_static_ratio"],
    },
    "tricycle": {
        "min_track_length": 60,
        "max_stationary_extent": 1.5,
        "required_signals": ["low_stationary_extent", "low_mean_speed", "high_static_ratio"],
    },
    "awning-tricycle": {
        "min_track_length": 60,
        "max_stationary_extent": 1.5,
        "required_signals": ["low_stationary_extent", "low_mean_speed", "high_static_ratio"],
    },
}

VRU_STATIC_GATE = {
    "pedestrian": {
        "min_track_length": 90,
        "max_stationary_extent": 1.0,
        "required_signals": ["low_stationary_extent", "low_mean_speed", "high_static_ratio"],
    },
    "people": {
        "min_track_length": 90,
        "max_stationary_extent": 1.0,
        "required_signals": ["low_stationary_extent", "low_mean_speed", "high_static_ratio"],
    },
    "bicycle": {
        "min_track_length": 60,
        "max_stationary_extent": 1.5,
        "required_signals": ["low_stationary_extent", "low_mean_speed", "high_static_ratio"],
    },
}
```

单位说明：

```text
displacement / path_length / stationary_extent / per_frame_motion_threshold: m
mean_speed / max_speed: m/s
```

当前静止判定分三层：

1. 标准车辆类：`car, truck, bus, freight_car, van`。继续使用原始车辆静止筛除逻辑：类别属于 `STATIC_GATE["filter_classes"]`，`numFrames >= STATIC_GATE["min_track_length"]`，且 `signals` 中包含 `low_stationary_extent`。
2. 轻型弱势车辆类：`motor, tricycle, awning-tricycle`。使用 `LIGHT_VRU_STATIC_GATE`，必须达到对应 `min_track_length`，并同时满足该类别自己的 `max_stationary_extent`、`low_mean_speed` 和 `high_static_ratio` 条件。`filter_reason` 以 `light_vru_static_gate` 开头。
3. 行人与自行车类：`pedestrian, people, bicycle`。使用 `VRU_STATIC_GATE`，其中 `pedestrian/people` 要求 `numFrames >= 90` 且 `stationary_extent <= 1.0`，`bicycle` 要求 `numFrames >= 60` 且 `stationary_extent <= 1.5`，并同时满足 `low_mean_speed` 和 `high_static_ratio`。`filter_reason` 以 `vru_static_gate` 开头。

标准车辆类中，`displacement <= max_displacement`、`mean_speed <= max_mean_speed`、`static_ratio >= static_ratio_threshold`、`path_length <= max_path_length` 会记录到 `filter_reason`，但不作为最终删除的必要条件。轻型弱势车辆类和行人与自行车类不会复用 `STATIC_GATE["max_stationary_extent"]`，而是直接使用各自配置中的 `max_stationary_extent` 与 `stationary_extent` 数值比较。

`stationary_extent` 使用轨迹点到中位中心点距离的 95% 分位数乘以 2，用于避免少量异常点污染首尾位移，也避免把逐帧检测抖动累计成很长的假路径。

### 疑似 ID 断裂过滤

疑似 ID 断裂过滤参数位于 `converter.py` 的 `FRAGMENTATION_FILTER`：

```python
FRAGMENTATION_FILTER = {
    "enabled": True,
    "min_gap_frames": 5,
    "max_gap_frames": 100,
    "min_track_frames": 5,
    "score_threshold": 0.70,
    "max_position_distance_base": 30.0,
    "max_position_distance_gap_factor": 2.0,
    "min_bbox_size_ratio": 0.4,
    "max_bbox_size_ratio": 2.5,
    "max_heading_diff_deg": 60.0,
    "border_margin_px": 20.0,
    "velocity_window_frames": 5,
    "filter_strategy": "drop_all_suspected_fragments",
}
```

该过滤只判断同一类别兼容组内、时间 gap 合理、预测位置接近、bbox 尺寸相近、运动方向合理且非明显边界进出的 raw tracklet。达到阈值后采用 `drop_all_suspected_fragments`：整组 raw tracklet 全部从 `moving_filtered` 删除。

## 十、补全、修复和平滑逻辑

### 1. 原始逐帧记录展开

对 `traj_info` 中每个目标行，先生成内部记录：

```text
frame
output_frame
object_id
category_id
raw_class
confidence
lane_id
q1_x ~ q4_y
world_q1_x ~ world_q4_y
xCenter_raw
yCenter_raw
raw_width
raw_length
is_interpolated
is_outlier
source_row_index
```

中心点计算：

```python
xCenter_raw = mean(world_q1_x, world_q2_x, world_q3_x, world_q4_x)
yCenter_raw = mean(world_q1_y, world_q2_y, world_q3_y, world_q4_y)
```

尺寸计算：

```python
e12 = distance(q1, q2)
e23 = distance(q2, q3)
e34 = distance(q3, q4)
e41 = distance(q4, q1)
edge_a = mean(e12, e34)
edge_b = mean(e23, e41)
raw_width = min(edge_a, edge_b)
raw_length = max(edge_a, edge_b)
```

如果世界坐标不是有限数，则该行跳过并记录统计。缺失 `object_id` 的行也会跳过并记录 warning。

### 2. 分组、去重与缺帧比例

先按原始 `object_id` 分组，再按 `frame` 排序。当前版本不再因为单个 gap 过大或局部速度异常而拆分轨迹；同一个 `object_id` 会作为一个整体评估。

如果同一 `object_id` 在同一帧出现多条记录，转换阶段会保留 `confidence` 最高的一条，并在质量报告中记录。

对每个原始 `object_id` 统计：

```python
min_frame = first observed frame
max_frame = last observed frame
expected_frame_count = max_frame - min_frame + 1
observed_frame_count = unique observed frame count
missing_frame_count = expected_frame_count - observed_frame_count
missing_ratio = missing_frame_count / expected_frame_count
```

当前阈值：

```python
MAX_MISSING_RATIO = 0.40
```

| 条件 | 当前处理 |
| --- | --- |
| `missing_ratio > 0.40` | 丢弃整条原始轨迹，不进入最终 CSV。`drop_reason = "missing_ratio_exceeded"`。 |
| `missing_ratio <= 0.40` | 保留整条轨迹，并补齐 `min_frame` 到 `max_frame` 之间所有缺失帧。 |

整体缺帧比例通过后，还会执行短轨迹过滤。该过滤发生在 `full` 输出生成之前，因此 `full`、`moving_filtered` 和 `Final Data` 都不会保留这些短暂目标：

| 类别 | 最小保留帧数 |
| --- | --- |
| `pedestrian` / `people` | 90 |
| 其他类别 | 150 |

### 3. 缺失帧补全

补全发生在同一保留轨迹内部相邻两条真实检测记录之间。轨迹是否保留只由整体 `missing_ratio` 决定，不再由单个 gap 长度决定。

缺失帧数：

```python
gap = frame_next - frame_prev - 1
```

| 条件 | 当前处理 |
| --- | --- |
| `gap <= 0` | 不补全。 |
| `gap > 0` 且该轨迹 `missing_ratio <= MAX_MISSING_RATIO` | 无论 gap 长短，都补齐中间所有缺失帧。 |
| 该轨迹 `missing_ratio > MAX_MISSING_RATIO` | 整条轨迹已在分组阶段丢弃，不进入补全。 |

补全字段处理：

| 字段 | 补全方法 |
| --- | --- |
| `frame` | 使用缺失帧号。 |
| `output_frame` | 当前代码设置为补全帧号。 |
| `xCenter_raw/yCenter_raw` | 对前后中心点插值。当前只有两端点，因此实际使用线性插值；如果可用点数达到 3 且 scipy 可用，函数支持 PCHIP。 |
| `raw_width/raw_length` | 对前后尺寸线性插值，并保持 `raw_length >= raw_width`。 |
| `lane_id` | 前后 lane 一致则使用该 lane；不同则按缺失帧更靠近前端或后端选择。无法判断时为 `-1`。 |
| `raw_class` | 使用该轨迹的最终众数类别。 |
| `category_id` | 设为空。 |
| `confidence` | 设为 NaN，CSV 中为空。 |
| `is_interpolated` | 设为 True。 |
| `source_row_index` | 设为 -1。 |

`heading`、速度和加速度不会直接补全，而是在补全、修复和平滑之后统一重新计算。

### 4. 异常跳点修复

当前修复的是孤立跳点。对轨迹内部第 `i` 帧：

```python
d_prev = distance(p[i], p[i-1])
d_next = distance(p[i+1], p[i])
d_bridge = distance(p[i+1], p[i-1])
```

如果：

1. `p[i-1] -> p[i]` 或 `p[i] -> p[i+1]` 的速度超过类别 `max_speed`；
2. 但 `p[i-1] -> p[i+1]` 的桥接速度不超过类别 `max_speed`；

则认为第 `i` 帧是孤立跳点。处理方式：

```text
is_outlier = True
xCenter_raw = NaN
yCenter_raw = NaN
```

随后 `_fill_nan_centers()` 会用该轨迹其他有效中心点对 NaN 中心进行插值修复。当前代码只修复中心点跳点，不直接改该帧的 lane、class、confidence，也不删除该帧。

### 5. 尺寸异常剔除与无向框修正

最终 `width/length` 不使用某一帧瞬时尺寸，而是轨迹级稳定尺寸。尺寸统计只基于补帧之前的真实检测帧，不使用插值补出来的记录重新计算均值。

某帧尺寸满足以下任一条件时，被视为尺寸异常，不参与轨迹级尺寸均值：

```python
raw_width > 2.5 * median_width
raw_width < 0.4 * median_width
raw_length > 2.5 * median_length
raw_length < 0.4 * median_length
```

尺寸异常只影响 `width/length` 的统计，不会因为尺寸异常单独删除该帧中心点。

无向框旋转修正规则：

| 类别 | 修正尺寸 |
| --- | --- |
| `pedestrian` | `width = length = max(raw_mean_width, raw_mean_height)`，以中心点为中心按 `heading` 旋转显示。 |
| 其他无向框目标 | `width = short_side`，`length = long_side`，其中 `long_side` 沿 `heading`，`short_side` 垂直 `heading`。 |

### 6. 中心点平滑

平滑对象只有：

```text
xCenter_raw
yCenter_raw
```

输出为：

```text
xCenter
yCenter
```

当前平滑配置：

```python
ENABLE_TRAJECTORY_SMOOTHING = True
SMOOTH_METHOD = "savgol"
SMOOTH_WINDOW = 15
VEHICLE_SMOOTH_WINDOW = 45
SMOOTH_POLYORDER = 3
```

机动车类使用 `VEHICLE_SMOOTH_WINDOW = 45`，其他类别使用 `SMOOTH_WINDOW = 15`。实际窗口会限制为不超过轨迹长度的最大奇数；窗口不足 5、scipy 不可用或平滑失败时，直接返回原值。

不直接平滑的字段：

```text
width
length
class
lane_id
confidence
heading
velocity
acceleration
```

其中 `heading/velocity/acceleration` 是在中心点平滑之后重新计算。

### 7. heading、速度和加速度重算

当前 heading 配置：

```python
ENABLE_HEADING_REFINEMENT = True
HEADING_SMOOTH_WINDOW = 5
MIN_DISPLACEMENT_FOR_HEADING = 0.05
LOW_SPEED_THRESHOLD = 0.2
MAX_HEADING_JUMP_DEG = 45.0
```

heading 基于平滑后的中心点和运动方向计算，必要时会使用稳定运动方向、上一帧有效方向或世界四点框长边方向作为 fallback。最终 heading 序列使用 sin/cos 方式做窗口平滑，避免 0°/360° 跨界时直接平均角度数值。

速度和加速度使用真实帧号对应的时间差：

```python
t = frame / frameRate
```

中间记录使用中心差分，首尾记录使用前向或后向差分。纵横向速度和加速度使用 heading 投影：

```python
theta = radians(heading)
lonVelocity = xVelocity * sin(theta) + yVelocity * cos(theta)
latVelocity = xVelocity * (-cos(theta)) + yVelocity * sin(theta)
lonAcceleration = xAcceleration * sin(theta) + yAcceleration * cos(theta)
latAcceleration = xAcceleration * (-cos(theta)) + yAcceleration * sin(theta)
```

## 十一、quality_report.json

`quality_report.json` 是辅助质量报告，不是正式三张 CSV。主要包含：

| 字段 | 内容 |
| --- | --- |
| `folderName` | 数据集文件夹名。 |
| `recordingId/locationId` | 文件夹名解析结果。 |
| `detectionPkl/stabilizationPkl` | 使用到的 pkl 文件路径。 |
| `videoInfo/outputInfo/frameRate` | 视频信息和输出帧率。 |
| `arrayColumnCounts` | `traj_info` 中 array 列数统计。 |
| `rawObjectCount` | 原始 `object_id` 数量。 |
| `finalTrackCount` | 当前版本最终轨迹数量。 |
| `classTrackCounts` | 当前版本各类别轨迹数量。 |
| `numVehicles/numVRUs` | 当前版本车辆与 VRU 轨迹数量。 |
| `orthoPxToMeter` | 像素到米比例估计值。 |
| `quality` | 类别跳变、短轨迹、重复帧、整体缺帧比例、丢弃轨迹、插值、异常点、尺寸异常、平滑参数等质量统计。 |
| `staticGate` | 静止门控参数、轻型弱势车辆静止参数、行人与自行车静止参数、每条轨迹运动统计、被过滤轨迹列表。 |
| `fragmentationFilter` | 疑似 ID 断裂过滤参数、策略、候选组、组内 raw ID、评分和删除数量。`full` 中 `use_fragmentation_filter=false`。 |

## 十二、可视化读取逻辑

当前 `Visualization/app/server.py` 默认扫描：

```text
Visualization/Final Data/
```

数据集列表中的版本为：

```text
final
```

前端接口主要包括：

```text
GET  /api/datasets
GET  /api/datasets/<dataset_id>/final/metadata
GET  /api/datasets/<dataset_id>/final/tracks
GET  /api/datasets/<dataset_id>/final/objects
GET  /api/datasets/<dataset_id>/final/frames
GET  /api/datasets/<dataset_id>/final/lanes
GET  /api/datasets/<dataset_id>/final/background
POST /api/scan?force=true|false
```

后端读取 `Final Data` 的三张正式 CSV，同时会从 `Visualization/Initial results/<folderName>/` 读取背景图、pkl 和 road config json，用于背景对齐、四角点生成和车道叠加。

背景图优先级：

```text
background_*.jpg / jpeg / png
first_frame_*.jpg / jpeg / png
```

车道几何识别规则：

```text
road
lane_*
laneline_*
drivingline*
```

如果没有 road config json，前端会提示没有可绘制的车道几何；目标详情和悬停提示仍会显示当前帧 `lane_id`。

## 十三、当前实现边界

1. 当前不会跨不同 `object_id` 做保守拼接，`stitched_track_count = 0`；疑似断裂只在 `moving_filtered` 中整组删除。
2. 当前只修复孤立跳点；连续多帧异常不触发轨迹拆分，只记录质量信息并由整体缺帧比例决定是否保留。
3. `PCHIP` 插值函数存在，但缺失帧补全时通常只有前后两个端点，因此实际多为线性插值。
4. `Adjusted results` 保留追溯字段和质量文件；`Final Data` 是正式精简导出，默认来自 `moving_filtered`。
5. `Final Data` 导出 `tracks.csv` 时会对局部 `lane_id = -1` 做邻域修正；该修正只作用于正式导出，不改变 `Adjusted results` 的完整版本。
