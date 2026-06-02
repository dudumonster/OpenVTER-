# OpenVTER 轨迹 CSV 转换与可视化

详细字段计算、补全、修复和平滑逻辑见：

```text
Visualization/DATASET_CONVERSION_DETAILS.md
```

项目独立环境 `OpenVTER` 的使用说明见：

```text
Visualization/OPENVTER_ENVIRONMENT.md
```

## 放入原始结果

把每个地点的项目结果文件夹放到：

```text
Visualization/Initial results/
```

例如：

```text
Visualization/Initial results/cao_qiao_001/
```

常见文件包括：

```text
background_*.jpg
first_frame_*.jpg
det_bbox_result_*.pkl
*_stab.pkl
tracking_output_*.mp4
```

CSV 转换主数据源是 `det_bbox_result_*.pkl` 里的 `traj_info`。`raw_det` 只作为辅助检查，`*_stab.pkl` 不参与轨迹动力学计算。

## 执行转换

转换全部数据集：

```powershell
python "Visualization\app\converter.py" --force
```

只转换某一个数据集：

```powershell
python "Visualization\app\converter.py" --datasets cao_qiao_001 --force
```

检查 pkl 结构：

```powershell
python "Visualization\app\converter.py" --inspect "Visualization\Initial results\cao_qiao_001\det_bbox_result_cao_qiao_001.pkl"
```

总日志写入：

```text
Visualization/logs/conversion.log
```

每个版本自己的日志写入对应输出目录下的 `conversion_log.txt`。

## 输出目录

每个地点会输出两个版本：

```text
Visualization/Adjusted results/<folderName>/
├── full/
│   ├── <folderName>_recordingMeta.csv
│   ├── <folderName>_tracksMeta.csv
│   ├── <folderName>_tracks.csv
│   ├── conversion_log.txt
│   └── quality_report.json
└── moving_filtered/
    ├── <folderName>_recordingMeta.csv
    ├── <folderName>_tracksMeta.csv
    ├── <folderName>_tracks.csv
    ├── conversion_log.txt
    └── quality_report.json
```

`Visualization/Adjusted results/<folderName>/` 根目录下只保留 `full` 和 `moving_filtered` 两个文件夹，不再直接保存 CSV、日志或报告文件。

## full 与 moving_filtered

`full` 保留所有清洗、补全、平滑后的轨迹，不删除长期静止目标。

`moving_filtered` 在 `full` 基础上剔除长期静止的机动车类目标，用于减少画面里停放车辆对轨迹查看的干扰。被过滤的目标只会从 `moving_filtered` 中删除，`full` 永远保留完整结果。

静止门控参数在 [converter.py](app/converter.py) 顶部的 `STATIC_GATE` 中配置：

```python
STATIC_GATE = {
    "min_track_length": 30,
    "max_displacement": 1.0,
    "max_path_length": 2.0,
    "max_stationary_extent": 2.0,
    "max_mean_speed": 0.2,
    "static_ratio_threshold": 0.8,
    "per_frame_motion_threshold": 0.05,
}
```

这些参数采用当前标准轨迹单位：位置为 m，速度为 m/s，逐帧位移阈值为 m/frame。过滤主要作用于 `car、truck、bus、freight_car、van、motor、tricycle、awning-tricycle`。
静止过滤会把检测抖动造成的微小变化也当作静止处理：只要车辆 95% 轨迹点都落在很小范围内，就会从 `moving_filtered` 删除；不再依赖容易被少量异常点污染的首尾位移或累计路径作为必要条件。`full` 版本仍保留这些静止车辆，但可视化时不绘制方向箭头，只显示目标框。

`quality_report.json` 会记录静止门控参数、每条 track 的运动统计、被过滤目标数量和原因。`conversion_log.txt` 也会记录被过滤的 `trackId`、类别、位移、累计路径、平均速度和静止比例。

## 补帧与无向框修正

当前转换按原始 `object_id` 的整体缺帧比例处理轨迹，不再按单个 gap 拆分。`MAX_MISSING_RATIO = 0.40`：超过阈值的轨迹整条丢弃；不超过阈值的轨迹会补齐 `min_frame` 到 `max_frame` 之间所有缺失帧。重复帧会保留最高置信度记录，并写入质量报告。

过短轨迹会在生成 `full` 前统一丢弃：`pedestrian/people` 少于 90 帧删除，其他类别少于 150 帧删除。因此两个输出版本都不会保留这些短暂目标。

无向框目标的最终 `width/length` 使用补帧前真实检测帧的稳定尺寸。`pedestrian` 使用稳定正方形框；其他无向框目标使用长边沿 `heading`、短边垂直 `heading` 的旋转框。可视化工具界面和控件不变，后端返回的四角点已使用修正尺寸与 heading 生成。

## 文件夹名称解析

例如：

```text
cao_qiao_001 -> recordingId = 001, locationId = cao_qiao
```

如果文件夹名称最后一段不是纯数字，则 `recordingId` 和 `locationId` 都暂时使用文件夹名，并在日志中给出 warning。

## pkl 结构

当前转换按真实解析到的结构读取：

```python
(frame_id, output_frame, array)
```

`traj_info` 中的 `array` 为 `N x 20`，列含义为：

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

最终轨迹分析优先使用 `world_q1_x ~ world_q4_y`，像素四点框主要用于估算 `orthoPxToMeter`。

## 三张标准 CSV

每个版本都输出以下三张 CSV，字段顺序固定。

`<folderName>_recordingMeta.csv`：

```text
recordingId,locationId,frameRate,numFrames,duration,numTracks,numVehicles,numVRUs,classTrackCounts,orthoPxToMeter
```

`<folderName>_tracksMeta.csv`：

```text
recordingId,trackId,initialFrame,finalFrame,numFrames,startXCenter,startYCenter,endXCenter,endYCenter,startLaneId,endLaneId,width,length,raw_mean_width,raw_mean_height,corrected_width,corrected_height,box_orientation_source,missing_ratio,class
```

`<folderName>_tracks.csv`：

```text
recordingId,trackId,lane_id,frame,trackLifetime,xCenter,yCenter,heading,width,length,raw_mean_width,raw_mean_height,corrected_width,corrected_height,box_orientation_source,is_interpolated,missing_ratio,xVelocity,yVelocity,xAcceleration,yAcceleration,lonVelocity,latVelocity,lonAcceleration,latAcceleration,centerX,centerY
```

其中 `width` 和 `length` 是该轨迹修正后的稳定宽度和长度，同一个 `trackId` 的宽度和长度保持稳定；`raw_mean_width/raw_mean_height` 记录补帧前真实检测帧尺寸统计，`corrected_width/corrected_height` 与可视化旋转框使用的尺寸同步；`centerX = xCenter`，`centerY = yCenter`。

## 启动当前可视化工具

后端：

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

前端会自动扫描 `Visualization/Adjusted results/<folderName>/full/` 和 `Visualization/Adjusted results/<folderName>/moving_filtered/`，并在数据集列表中显示为：

```text
<folderName> / full
<folderName> / moving_filtered
```

如果页面显示“没有已转换数据集”，请先确认后端已经重启，并确认这两个版本目录下存在 `<folderName>_recordingMeta.csv`、`<folderName>_tracksMeta.csv`、`<folderName>_tracks.csv`。

选择数据集后，前端会通过后端自动读取背景图：优先使用 `Visualization/Initial results/<folderName>/background_*.jpg`，如果没有则使用 `first_frame_*.jpg`。标准 CSV 中的 world/meter 坐标会由后端结合 pkl 里的像素四点框和 world 四点框估计 `world -> pixel` 映射，再返回给前端绘制，从而与背景图对齐。

可视化中的 `显示车道` 开关会尝试读取真实车道几何并叠加到画布上。后端会优先在 `Visualization/Initial results/<folderName>/` 下查找 road_config json，识别其中的 `road`、`lane_*`、`laneline_*`、`drivingline*` 形状；如果背景图可用，road_config 中的像素几何会直接叠加到背景图坐标系。

注意：`det_bbox_result_*.pkl` 当前只保存了每帧目标的 `lane_id`，没有保存车道线或车道区域几何本身。因此如果结果文件夹里没有对应 road_config json，前端会提示未找到车道几何。目标详情和悬停提示始终会显示当前帧 `lane_id`。
