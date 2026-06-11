# Visualization 项目流程介绍

本文说明 `D:\OpenVTER-\Visualization` 的数据来源、转换流程、输出结构、字段约定和可视化读取逻辑。常用运行命令见 `VISUALIZATION_COMMANDS.md`。

## 1. 目录职责

```text
Visualization/
├─ Initial results/       输入目录：原始检测、跟踪、背景图和车道配置
├─ Adjusted results/      中间输出：完整结果、质量报告和过滤报告
├─ Final Data/            正式输出：给可视化和下游使用的精简 CSV
├─ Trajs Check/           轨迹运动学检查脚本
├─ logs/                  转换和服务日志
└─ app/                   转换器、后端服务和前端静态资源
```

`README.md` 是入口说明，不计入专题文档。专题文档只保留本文和 `VISUALIZATION_COMMANDS.md`。

## 2. 输入数据

每个数据集放在：

```text
Visualization\Initial results\<folderName>\
```

常见文件：

```text
det_bbox_result_*.pkl
*_stab.pkl
background_*.jpg / jpeg / png
first_frame_*.jpg / jpeg / png
tracking_output_*.mp4
road_config*.json 或其他包含 road/lane/laneline/drivingline 的 json
```

正式轨迹数据源是 `det_bbox_result_*.pkl` 中的 `traj_info`。`raw_det` 只作为辅助检查字段，`*_stab.pkl` 只记录为稳像文件，不参与最终轨迹运动学计算。

`traj_info` 中每帧通常是：

```python
(frame_id, output_frame, array)
```

或：

```python
(frame_id, output_frame, array, frame_time)
```

当前转换器按 `array` 的前 20 列解析：

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

像素四点框主要用于估计 `orthoPxToMeter` 和前端 world/pixel 映射。轨迹中心、尺寸、速度、加速度主要基于世界坐标四点框计算。

## 3. 转换总流程

转换入口是：

```text
Visualization\app\converter.py
```

主要流程：

1. 读取 `Initial results/<folderName>/det_bbox_result_*.pkl`。
2. 展开 `traj_info` 中逐帧目标记录。
3. 按原始 `object_id` 分组，去除同帧重复记录。
4. 统计缺帧比例，丢弃整体缺帧比例过高的轨迹。
5. 对保留轨迹补齐缺失帧。
6. 修复孤立跳点。
7. 统计轨迹级稳定尺寸，修正无向框宽高。
8. 平滑中心点。
9. 基于平滑中心点重算 heading、速度和加速度。
10. 生成 `Adjusted results` 的 `full` 版本。
11. 在 `full` 基础上做疑似 ID 断裂过滤和静止轨迹过滤，生成 `moving_filtered`。
12. 从 `moving_filtered` 导出 `Final Data` 的三张正式 CSV。

当前正式导出来源由转换器常量控制：

```python
FINAL_DATA_SOURCE_VERSION = "moving_filtered"
```

## 4. 输出结构

### Adjusted results

每个数据集输出两个版本：

```text
Visualization\Adjusted results\<folderName>\
├─ full\
└─ moving_filtered\
```

两个版本目录中都包含：

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

`full` 保留通过基础清洗、补全、修复和平滑后的完整轨迹。`moving_filtered` 在 `full` 基础上删除高置信疑似 ID 断裂组和长期静止轨迹。过滤不会反向修改 `full`。

### Final Data

正式输出只保留三张精简 CSV：

```text
Visualization\Final Data\<folderName>\
├─ <folderName>_recordingMeta.csv
├─ <folderName>_tracksMeta.csv
└─ <folderName>_tracks.csv
```

`Final Data` 是可视化后端默认扫描和读取的数据目录。

## 5. 核心 CSV 字段

### recordingMeta.csv

```text
recordingId,locationId,frameRate,numFrames,duration,numTracks,numVehicles,numVRUs,classTrackCounts,orthoPxToMeter
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `recordingId` | 从数据集文件夹名解析。比如 `cao_qiao_001` 得到 `001`。 |
| `locationId` | 从数据集文件夹名解析。比如 `cao_qiao_001` 得到 `cao_qiao`。 |
| `frameRate` | 优先读取 pkl 中的输出帧率，缺失时使用 `29.97`。 |
| `numFrames` | 优先读取视频总帧数，缺失时根据 `traj_info` 推断。 |
| `duration` | `numFrames / frameRate`。 |
| `numTracks` | 当前版本最终保留的轨迹数量。 |
| `numVehicles` | 机动车类轨迹数量。 |
| `numVRUs` | 弱势交通参与者轨迹数量。 |
| `classTrackCounts` | 各类别轨迹数量 JSON 字符串。 |
| `orthoPxToMeter` | 基于像素框边长与世界框边长估计的像素到米比例。 |

### tracksMeta.csv

`Adjusted results` 完整字段：

```text
recordingId,trackId,raw_object_id,initialFrame,finalFrame,numFrames,startXCenter,startYCenter,endXCenter,endYCenter,startLaneId,endLaneId,width,length,raw_mean_width,raw_mean_height,corrected_width,corrected_height,box_orientation_source,missing_ratio,class
```

`Final Data` 精简字段：

```text
recordingId,trackId,initialFrame,finalFrame,numFrames,startXCenter,startYCenter,endXCenter,endYCenter,startLaneId,endLaneId,width,length,class
```

### tracks.csv

`Adjusted results` 完整字段：

```text
recordingId,trackId,raw_object_id,lane_id,frame,trackLifetime,xCenter,yCenter,heading,width,length,raw_mean_width,raw_mean_height,corrected_width,corrected_height,box_orientation_source,is_interpolated,missing_ratio,xVelocity,yVelocity,xAcceleration,yAcceleration,lonVelocity,latVelocity,lonAcceleration,latAcceleration
```

`Final Data` 精简字段：

```text
recordingId,trackId,lane_id,frame,trackLifetime,xCenter,yCenter,heading,width,length,xVelocity,yVelocity,xAcceleration,yAcceleration,lonVelocity,latVelocity,lonAcceleration,latAcceleration
```

`Final Data` 会去掉追溯和辅助字段，例如 `raw_object_id`、`raw_mean_width`、`corrected_width`、`is_interpolated`、`missing_ratio` 等。排查质量问题时应回到 `Adjusted results` 查看完整 CSV、`id_mapping.csv`、`filter_report.csv` 和 `quality_report.json`。

## 6. 补全、修复和平滑逻辑

### 缺帧比例

每个原始 `object_id` 按以下方式统计：

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

`missing_ratio > 0.40` 的轨迹会整条丢弃；其余轨迹会补齐 `min_frame` 到 `max_frame` 之间的缺失帧。

### 短轨迹过滤

短轨迹过滤发生在 `full` 输出生成之前：

| 类别 | 最少保留帧数 |
| --- | --- |
| `pedestrian` / `people` | 90 |
| 其他类别 | 150 |

### 缺失帧补全

缺失帧在同一保留轨迹内部、相邻两条真实检测记录之间补全。中心点和尺寸使用插值；类别使用轨迹最终众数类别；`confidence` 为空；`is_interpolated=True`。`heading`、速度和加速度不直接补全，而是在后续统一重算。

### 孤立跳点修复

对轨迹内部单帧跳点，如果前后两段速度异常、但跨过该点后的桥接速度合理，则将该帧中心点置为 NaN，再用轨迹内其他有效中心点插值修复。当前只修复孤立跳点，不因连续异常自动拆分轨迹。

### 中心点平滑

平滑对象：

```text
xCenter_raw
yCenter_raw
```

输出：

```text
xCenter
yCenter
```

当前配置：

```python
ENABLE_TRAJECTORY_SMOOTHING = True
SMOOTH_METHOD = "savgol"
SMOOTH_WINDOW = 15
VEHICLE_SMOOTH_WINDOW = 45
SMOOTH_POLYORDER = 3
```

机动车类使用 `VEHICLE_SMOOTH_WINDOW = 45`，其他类别使用 `SMOOTH_WINDOW = 15`。

## 7. Heading、速度和加速度

heading 定义：

```text
0 deg   -> +Y
90 deg  -> +X
180 deg -> -Y
270 deg -> -X
```

当前 heading 配置：

```python
ENABLE_HEADING_REFINEMENT = True
HEADING_SMOOTH_WINDOW = 5
MIN_DISPLACEMENT_FOR_HEADING = 0.05
LOW_SPEED_THRESHOLD = 0.2
MAX_HEADING_JUMP_DEG = 45.0
```

heading 基于平滑后的中心点和运动方向计算。低速或位移不足时，会使用稳定运动方向、上一帧有效方向或 fallback 方向，避免厘米级抖动导致朝向乱跳。最终 heading 序列使用 sin/cos 方式做角度平滑，避免 0/360 度跨界时直接平均角度数值。

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

## 8. 过滤逻辑

### 静止过滤

标准机动车类：

```text
car, truck, bus, freight_car, van
```

轻型弱势车辆类：

```text
motor, tricycle, awning-tricycle
```

行人与自行车类：

```text
pedestrian, people, bicycle
```

静止判定会结合轨迹长度、平均速度、静止比例、路径长度和 `stationary_extent` 等指标。不同类别使用不同阈值，避免把轻型车和行人的自然小范围移动误判为标准车辆的静止状态。

### 疑似 ID 断裂过滤

`FRAGMENTATION_FILTER` 只判断同类别兼容组内、时间 gap 合理、预测位置接近、bbox 尺寸相近、运动方向合理且非明显边界进出的 raw tracklet。达到阈值后使用：

```text
drop_all_suspected_fragments
```

也就是整组疑似断裂 raw tracklet 从 `moving_filtered` 删除。当前不会做 tracklet stitching，也不会把多个 raw ID 合并成同一个最终 ID。

## 9. 质量追溯文件

`Adjusted results/<folderName>/<version>/quality_report.json` 是辅助质量报告，不属于正式三张 CSV。它主要记录：

```text
输入 pkl 信息
array 列数统计
原始 object 数量
最终 track 数量
类别统计
缺帧、补帧、重复帧、短轨迹、跳点、尺寸异常
平滑参数
静止过滤统计
疑似 ID 断裂过滤统计
```

`id_mapping.csv` 记录原始 `raw_object_id` 到最终 `trackId/object_id` 的映射。被过滤的 raw tracklet 也会保留记录，并写明过滤原因。

`filter_report.csv` 记录未进入当前版本的 raw tracklet 以及对应过滤原因。

## 10. 可视化读取逻辑

后端入口：

```text
Visualization\app\server.py
```

默认扫描：

```text
Visualization\Final Data\
```

主要接口：

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

后端读取 `Final Data` 的三张正式 CSV，同时从 `Initial results/<folderName>/` 读取背景图、pkl 和 road config json，用于背景对齐、四角点生成和车道叠加。

背景图优先级：

```text
background_*.jpg / jpeg / png
first_frame_*.jpg / jpeg / png
```

车道几何识别关键字：

```text
road
lane_*
laneline_*
drivingline*
```

如果没有 road config json，前端会提示没有可绘制的车道几何；目标详情和悬停提示仍会显示当前帧的 `lane_id`。
