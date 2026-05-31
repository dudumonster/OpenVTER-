# 数据集转换结果与轨迹清洗逻辑说明

本文档说明当前 `Visualization/app/converter.py` 的真实转换逻辑，重点包括：

1. 输出目录、输出文件，以及三张 CSV 的字段含义和计算方法。
2. 当前项目中轨迹补全、异常修复和平滑的具体处理方式。

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

像素四点框 `q1_x ~ q4_y` 主要用于估算 `orthoPxToMeter`。轨迹中心、尺寸、速度、加速度主要基于世界坐标四点框 `world_q1_x ~ world_q4_y` 计算。

## 二、输出目录与版本

每个原始地点文件夹会输出两个版本：

```text
Visualization/Adjusted results/<folderName>/
├── full/
└── moving_filtered/
```

根目录 `Visualization/Adjusted results/<folderName>/` 不再直接保存 CSV、日志或报告文件，只保留 `full` 和 `moving_filtered` 两个子文件夹。

### full

完整版本。保留所有清洗、补全、平滑后的最终轨迹。

### moving_filtered

运动过滤版本。在 `full` 的基础上剔除长期静止的机动车类轨迹。该版本只删除满足静止门控条件的最终 `trackId`，不会影响 `full` 版本。

静止过滤参数位于 `converter.py` 的 `STATIC_GATE`：

```python
STATIC_GATE = {
    "min_track_length": 30,
    "max_displacement": 1.0,
    "max_mean_speed": 0.2,
    "static_ratio_threshold": 0.8,
    "per_frame_motion_threshold": 0.05,
    "filter_classes": sorted(VEHICLE_CLASSES),
}
```

单位说明：

```text
displacement / path_length / per_frame_motion_threshold: m
mean_speed / max_speed: m/s
```

静止判定逻辑：

1. 轨迹类别必须属于 `filter_classes`。
2. `numFrames >= min_track_length`。
3. 以下三个静止信号至少满足两个：
   - `displacement <= max_displacement`
   - `mean_speed <= max_mean_speed`
   - `static_ratio >= static_ratio_threshold`

`static_ratio` 的计算方式是：相邻记录中心点位移小于等于 `per_frame_motion_threshold` 的比例。

`moving_filtered` 会重新给保留轨迹分配连续 `trackId`，从 1 开始。

## 三、每个版本的输出文件

每个版本目录下由当前标准转换模块输出：

```text
<folderName>_recordingMeta.csv
<folderName>_tracksMeta.csv
<folderName>_tracks.csv
conversion_log.txt
quality_report.json
```

如果目录中还存在旧版前端使用的 `tracks.csv、objects.csv、frames.csv、metadata.json、background.jpg`，它们属于历史可视化兼容产物，不是当前标准三张 CSV 转换逻辑的正式输出。

## 四、recordingMeta.csv 字段

字段顺序固定：

```text
recordingId,locationId,frameRate,numFrames,duration,numTracks,numVehicles,numVRUs,classTrackCounts,orthoPxToMeter
```

字段说明：

| 字段 | 计算逻辑 |
| --- | --- |
| `recordingId` | 从文件夹名解析。若 `cao_qiao_001`，最后一段 `001` 是编号，则 `recordingId = 001`。 |
| `locationId` | 从文件夹名解析。若 `cao_qiao_001`，则 `locationId = cao_qiao`。如果文件夹名不符合最后一段为数字的规则，则二者都使用文件夹名，并写 warning。 |
| `frameRate` | 优先读取 `output_info["output_fps"]`，如果没有则使用 `29.97`。 |
| `numFrames` | 优先读取 `video_info[0]["total_frames"]`。如果没有，则根据 `traj_info` 帧号推断。 |
| `duration` | `numFrames / frameRate`，单位秒。 |
| `numTracks` | 当前版本最终轨迹数量，按唯一最终 `trackId` 计数。`full` 和 `moving_filtered` 可能不同。 |
| `numVehicles` | 按最终 `class` 统计机动车轨迹数量。类别集合：`car, truck, bus, freight_car, van, motor, tricycle, awning-tricycle`。 |
| `numVRUs` | 按最终 `class` 统计弱势交通参与者轨迹数量。类别集合：`pedestrian, people, bicycle, tricycle, awning-tricycle, motor`。`motor/tricycle/awning-tricycle` 会同时计入车辆和 VRU。 |
| `classTrackCounts` | JSON 字符串，记录所有类别的最终轨迹数量。所有类别都会出现，即使数量为 0。 |
| `orthoPxToMeter` | 基于像素四点框边长和世界四点框边长的比例估算。对所有有效边计算 `world_edge_length / pixel_edge_length`，过滤 IQR 异常后取中位数。有效边少于 30 时输出空值并写 warning。 |

## 五、tracksMeta.csv 字段

字段顺序固定：

```text
recordingId,trackId,initialFrame,finalFrame,numFrames,startXCenter,startYCenter,endXCenter,endYCenter,startLaneId,endLaneId,width,length,class
```

字段说明：

| 字段 | 计算逻辑 |
| --- | --- |
| `recordingId` | 同 `recordingMeta.csv`。 |
| `trackId` | 当前版本内重新编号，从 1 开始连续编号。排序依据为轨迹第一帧和原始 `object_id`。 |
| `initialFrame` | 当前最终轨迹输出记录中的最小 `frame`。如果有补全帧，补全帧也参与最小值计算。 |
| `finalFrame` | 当前最终轨迹输出记录中的最大 `frame`。 |
| `numFrames` | 当前最终轨迹实际输出到 `tracks.csv` 的记录数，包括插值补全帧。不是简单的 `finalFrame - initialFrame + 1`。 |
| `startXCenter` | 该轨迹第一条输出记录的平滑后 `xCenter`。 |
| `startYCenter` | 该轨迹第一条输出记录的平滑后 `yCenter`。 |
| `endXCenter` | 该轨迹最后一条输出记录的平滑后 `xCenter`。 |
| `endYCenter` | 该轨迹最后一条输出记录的平滑后 `yCenter`。 |
| `startLaneId` | 第一条输出记录的 `lane_id`。无法判断时为 `-1`。 |
| `endLaneId` | 最后一条输出记录的 `lane_id`。无法判断时为 `-1`。 |
| `width` | 轨迹级稳定宽度，单位 m。来自该轨迹有效 `raw_width` 的均值；如果有效尺寸不足，则依次 fallback 到轨迹尺寸中位数、同类别平均尺寸、空值。 |
| `length` | 轨迹级稳定长度，单位 m。来自该轨迹有效 `raw_length` 的均值；fallback 逻辑同 `width`。保证 `length >= width`。 |
| `class` | 轨迹最终类别。按该最终轨迹真实检测帧的 `raw_class` 众数确定，不统计插值帧。 |

类别众数规则：

1. 只统计非插值帧。
2. 出现次数最多的类别作为最终 `class`。
3. 如果次数相同，则比较这些类别的 `confidence` 总和，较高者胜出。
4. 如果次数和置信度总和仍相同，则取最早出现的类别，并写 warning。
5. 如果最终类别占比 `< 0.7`，记录为类别不稳定轨迹，但不删除。

## 六、tracks.csv 字段

字段顺序固定：

```text
recordingId,trackId,lane_id,frame,trackLifetime,xCenter,yCenter,heading,width,length,xVelocity,yVelocity,xAcceleration,yAcceleration,lonVelocity,latVelocity,lonAcceleration,latAcceleration,centerX,centerY
```

字段说明：

| 字段 | 计算逻辑 |
| --- | --- |
| `recordingId` | 同 `recordingMeta.csv`。 |
| `trackId` | 与 `tracksMeta.csv` 中的最终 `trackId` 对应。 |
| `lane_id` | 原始 `traj_info` 第 19 列。补全帧会根据前后帧 lane 填充。无法判断时为 `-1`。 |
| `frame` | 原始帧号，不重新从 0 或 1 编号。补全帧使用对应缺失帧号。 |
| `trackLifetime` | 当前最终轨迹内第几条输出记录，从 1 开始递增。按实际输出记录数递增，不用 `frame - initialFrame + 1`。 |
| `xCenter` | 世界四点框中心点经过异常修复、插值补全、Savitzky-Golay 平滑后的 x 坐标，单位 m。 |
| `yCenter` | 世界四点框中心点经过异常修复、插值补全、Savitzky-Golay 平滑后的 y 坐标，单位 m。 |
| `heading` | 基于平滑后的中心点计算，单位 deg。定义为 `0° -> +Y, 90° -> +X, 180° -> -Y, 270° -> -X`。 |
| `width` | 该 `trackId` 的轨迹级稳定宽度，所有帧相同，单位 m。 |
| `length` | 该 `trackId` 的轨迹级稳定长度，所有帧相同，单位 m。 |
| `xVelocity` | 基于平滑后 `xCenter` 对真实时间差分得到，单位 m/s。 |
| `yVelocity` | 基于平滑后 `yCenter` 对真实时间差分得到，单位 m/s。 |
| `xAcceleration` | 基于 `xVelocity` 对真实时间差分得到，单位 m/s²。 |
| `yAcceleration` | 基于 `yVelocity` 对真实时间差分得到，单位 m/s²。 |
| `lonVelocity` | 沿 heading 前进方向的纵向速度，单位 m/s。 |
| `latVelocity` | 垂直 heading 的横向速度，单位 m/s。正值表示目标向自身左侧运动。 |
| `lonAcceleration` | 沿 heading 前进方向的纵向加速度，单位 m/s²。 |
| `latAcceleration` | 垂直 heading 的横向加速度，单位 m/s²。 |
| `centerX` | 兼容字段，当前等于 `xCenter`。 |
| `centerY` | 兼容字段，当前等于 `yCenter`。 |

### heading 计算

使用平滑后的中心点 `xCenter/yCenter`。对第 `i` 条记录：

```python
half_window = round(0.25 * frameRate)
j0 = max(0, i - half_window)
j1 = min(n - 1, i + half_window)
dx = xSmooth[j1] - xSmooth[j0]
dy = ySmooth[j1] - ySmooth[j0]
```

如果：

```python
sqrt(dx * dx + dy * dy) >= 0.2
```

则：

```python
heading = degrees(atan2(dx, dy)) % 360
```

如果位移不足 0.2 m，则沿用上一帧有效 heading。若该轨迹前面也没有有效 heading，则使用世界四点框长边方向作为 fallback；仍无法计算时填 `0.0` 并写 warning。

### 速度与加速度

时间戳使用：

```python
t = frame / frameRate
```

中间记录使用中心差分，首尾记录使用前向或后向差分。存在跳帧时，使用真实帧号对应的时间差，不假设相邻输出记录一定间隔 1 帧。

纵横向速度和加速度使用 heading 投影：

```python
theta = radians(heading)
lonVelocity = xVelocity * sin(theta) + yVelocity * cos(theta)
latVelocity = xVelocity * (-cos(theta)) + yVelocity * sin(theta)
lonAcceleration = xAcceleration * sin(theta) + yAcceleration * cos(theta)
latAcceleration = xAcceleration * (-cos(theta)) + yAcceleration * sin(theta)
```

## 七、quality_report.json

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
| `quality` | 类别跳变、短轨迹、跳帧、插值、异常点、尺寸异常等质量统计。 |
| `staticGate` | 静止门控参数、每条轨迹运动统计、被过滤轨迹列表。 |

## 八、conversion_log.txt

`conversion_log.txt` 记录转换过程中的关键日志，包括：

```text
输入文件夹
recordingId / locationId
读取到的 pkl 文件
video_info 与 frameRate
原始 object_id 数量
最终 trackId 数量
类别轨迹统计
静止过滤数量和被过滤 trackId
类别跳变轨迹
lane_id 为 -1 的数量
跳帧、补全、异常点、拆分、尺寸异常等统计
```

## 九、补全、修复和平滑逻辑

本节单独说明当前项目对轨迹数据的处理流程。

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

### 2. 原始轨迹分组与拆分

先按原始 `object_id` 分组，再按 `frame` 排序。每个原始组会根据以下规则拆分成轨迹片段：

1. 相邻记录缺失帧数 `gap > LONG_GAP_SPLIT`，当前值为 30，则拆分。
2. 相邻两点所需速度超过该类别 `max_speed * 1.5`，则拆分。

类别物理速度阈值来自 `PHYSICAL_LIMITS`，例如：

```text
car/truck/bus/van/freight_car: max_speed = 25.0 m/s
motor: 20.0 m/s
bicycle/tricycle/awning-tricycle: 12.0 m/s
pedestrian/people: 6.0 m/s
```

注意：当前代码没有执行跨不同 `object_id` 的保守拼接，`stitched_track_count` 固定记录为 0。

### 3. 类别修正

每个轨迹片段会重新计算最终类别：

```text
最终 class = 非插值真实检测帧 raw_class 的众数类别
```

插值帧不参与类别众数统计。类别跳变和类别不稳定只记录质量信息，不直接删除轨迹。

### 4. 异常跳点修复

当前修复的是“孤立跳点”。对轨迹内部第 `i` 帧：

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

随后 `_fill_nan_centers()` 会用该轨迹其他有效中心点对 NaN 中心进行插值修复。

当前代码只修复中心点跳点，不直接改该帧的 lane、class、confidence，也不删除该帧。

### 5. 缺失帧补全

补全发生在同一轨迹片段内部相邻两条记录之间。

缺失帧数：

```python
gap = frame_next - frame_prev - 1
```

处理规则：

| gap 范围 | 当前处理 |
| --- | --- |
| `gap <= 0` | 不补全。 |
| `1 <= gap <= 5` | 直接补全。 |
| `6 <= gap <= 15` | 只有当前后位置速度不超过类别 `max_speed`，且 lane 相同或至少一端 lane 为 `-1` 时补全。 |
| `gap > 15` | 不补全。 |

补全字段处理：

| 字段 | 补全方法 |
| --- | --- |
| `frame` | 使用缺失帧号。 |
| `output_frame` | 当前代码设置为补全帧号。 |
| `xCenter_raw/yCenter_raw` | 对前后中心点插值。当前只有两端点，因此实际使用线性插值；如果可用点数达到 3 且 scipy 可用，函数支持 PCHIP。 |
| `raw_width/raw_length` | 对前后尺寸线性插值，并保持 `raw_length >= raw_width`。 |
| `lane_id` | 前后 lane 一致则使用该 lane；不同则按缺失帧更靠近前端或后端选择。无法判断时为 `-1`。 |
| `raw_class` | 使用该轨迹片段的最终众数类别。 |
| `category_id` | 设为空。 |
| `confidence` | 设为 NaN，CSV 中为空。 |
| `is_interpolated` | 设为 True。 |
| `source_row_index` | 设为 -1。 |

不直接补全或插值的字段：

```text
heading
xVelocity
yVelocity
xAcceleration
yAcceleration
lonVelocity
latVelocity
lonAcceleration
latAcceleration
```

这些字段会在补全、修复和平滑之后统一重新计算。

### 6. 尺寸异常剔除

最终 `width/length` 不使用某一帧瞬时尺寸，而是轨迹级稳定尺寸。

先对该轨迹所有有效 `raw_width/raw_length` 计算中位数：

```python
median_width
median_length
```

某帧尺寸满足以下任一条件时，被视为尺寸异常，不参与轨迹级尺寸均值：

```python
raw_width > 2.5 * median_width
raw_width < 0.4 * median_width
raw_length > 2.5 * median_length
raw_length < 0.4 * median_length
```

尺寸异常只影响 `width/length` 的统计，不会因为尺寸异常单独删除该帧中心点。

最终尺寸计算：

1. 优先使用有效 `raw_width/raw_length` 的均值。
2. 如果有效尺寸不足，使用该轨迹尺寸中位数。
3. 如果仍不足，使用同类别轨迹尺寸均值。
4. 如果仍无法计算，填空值并写 warning。

### 7. 中心点平滑

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

默认使用 Savitzky-Golay：

```python
window_length = 15
polyorder = 2
mode = "interp"
```

短轨迹处理：

| 轨迹长度 | 处理 |
| --- | --- |
| `n >= 15` | 使用窗口 15。 |
| `5 <= n < 15` | 使用不超过轨迹长度的最大奇数窗口。 |
| `n < 5` | 不平滑，直接使用修复/补全后的中心点，并记录短轨迹。 |
| scipy 不可用 | 不平滑，直接返回原值。 |

不平滑的字段：

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

其中 `heading/velocity/acceleration` 是在中心点平滑之后重新计算，不是直接平滑原始值。

### 8. 速度、加速度和 heading 重算

补全、异常点修复、中心点平滑完成后，再计算：

```text
heading
xVelocity
yVelocity
xAcceleration
yAcceleration
lonVelocity
latVelocity
lonAcceleration
latAcceleration
```

因此这些动力学字段不会使用原始检测框中心点直接差分，也不会保留补全前的旧值。

### 9. full 到 moving_filtered 的过滤

`moving_filtered` 不重新做清洗、补全和平滑，而是基于 `full` 已经生成的最终 `tracksMeta/tracks` 计算每条轨迹的运动指标：

```text
displacement
path_length
mean_speed
max_speed
static_ratio
```

判定为静止的轨迹从 `moving_filtered` 删除。删除后，保留轨迹重新编号为连续 `trackId`，并同步更新 `tracksMeta.csv` 和 `tracks.csv`。

## 十、当前实现边界

需要特别注意当前代码的实际边界：

1. 当前不会跨不同 `object_id` 做保守拼接，`stitched_track_count = 0`。
2. 当前只修复孤立跳点；连续多帧异常主要通过拆分和日志记录处理。
3. `PCHIP` 插值函数存在，但缺失帧补全时通常只有前后两个端点，因此实际多为线性插值。
4. `tracksMeta.csv` 中已取消 `meanWidth/meanLength` 字段；`width/length` 就是轨迹级稳定平均尺寸。
5. `moving_filtered` 只是额外过滤版本，不会修改 `full` 的完整结果。
