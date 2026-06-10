# OpenVTER 轨迹 CSV 转换与可视化

本目录用于把 OpenVTER 的 `det_bbox_result_*.pkl` 结果转换为标准轨迹 CSV，并通过本地网页工具查看轨迹、目标框、类别、车道和背景图对齐效果。

更详细的字段计算、补全、修复、过滤和平滑逻辑见：

```text
Visualization/DATASET_CONVERSION_DETAILS.md
```

独立 conda 环境说明见：

```text
Visualization/OPENVTER_ENVIRONMENT.md
```

## 目录结构

```text
Visualization/
├── Initial results/        # 输入：每个数据集的原始结果文件夹
├── Adjusted results/       # 中间输出：full 与 moving_filtered 两个完整版本
├── Final Data/             # 正式输出：从 moving_filtered 导出的三张精简 CSV
├── logs/                   # 转换和可视化日志
└── app/
    ├── converter.py        # 转换入口
    ├── server.py           # 本地可视化后端
    └── static/             # 前端页面
```

## 放入原始结果

把每个地点或视频的结果文件夹放到：

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
road_config*.json 或其他包含 road/lane/laneline/drivingline 标注的 json
```

转换主数据源是 `det_bbox_result_*.pkl` 中的 `traj_info`。`raw_det` 只作为辅助检查字段，`*_stab.pkl` 只记录为稳像文件，不参与轨迹动力学计算。

## 执行转换

在项目根目录 `D:\OpenVTER-` 下运行。

转换全部数据集：

```powershell
python "Visualization\app\converter.py" --force
```

只转换一个或多个数据集：

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

每个版本自己的转换日志写入对应版本目录下的 `conversion_log.txt`。

## 输出结果

转换后会生成两类输出。

### 1. Adjusted results

`Adjusted results` 保留完整追溯字段、质量报告和过滤报告。每个数据集有两个版本：

```text
Visualization/Adjusted results/<folderName>/
├── full/
│   ├── <folderName>_recordingMeta.csv
│   ├── <folderName>_tracksMeta.csv
│   ├── <folderName>_tracks.csv
│   ├── id_mapping.csv
│   ├── filter_report.csv
│   ├── metadata.json
│   ├── conversion_log.txt
│   └── quality_report.json
└── moving_filtered/
    ├── <folderName>_recordingMeta.csv
    ├── <folderName>_tracksMeta.csv
    ├── <folderName>_tracks.csv
    ├── id_mapping.csv
    ├── filter_report.csv
    ├── metadata.json
    ├── conversion_log.txt
    └── quality_report.json
```

`full` 保留所有清洗、补全、修复和平滑后的轨迹。`moving_filtered` 在 `full` 基础上先删除高置信疑似 ID 断裂组，再删除长期静止的机动车类轨迹。过滤只影响 `moving_filtered`，不会修改 `full`。

### 2. Final Data

`Final Data` 是当前正式数据出口，也是可视化工具当前默认读取的数据目录。它从 `moving_filtered` 导出，只保留三张精简标准 CSV：

```text
Visualization/Final Data/<folderName>/
├── <folderName>_recordingMeta.csv
├── <folderName>_tracksMeta.csv
└── <folderName>_tracks.csv
```

`Final Data` 默认来源由 `converter.py` 中的 `FINAL_DATA_SOURCE_VERSION = "moving_filtered"` 控制。

## CSV 字段

`Adjusted results/<folderName>/<version>/<folderName>_tracks.csv` 字段为：

```text
recordingId,trackId,raw_object_id,lane_id,frame,trackLifetime,xCenter,yCenter,heading,width,length,raw_mean_width,raw_mean_height,corrected_width,corrected_height,box_orientation_source,is_interpolated,missing_ratio,xVelocity,yVelocity,xAcceleration,yAcceleration,lonVelocity,latVelocity,lonAcceleration,latAcceleration
```

`Final Data/<folderName>/<folderName>_tracks.csv` 字段为：

```text
recordingId,trackId,lane_id,frame,trackLifetime,xCenter,yCenter,heading,width,length,xVelocity,yVelocity,xAcceleration,yAcceleration,lonVelocity,latVelocity,lonAcceleration,latAcceleration
```

`Final Data` 会去掉追溯和辅助字段，例如 `raw_object_id`、`raw_mean_width`、`corrected_width`、`is_interpolated`、`missing_ratio` 等。需要排查转换质量或追溯原始 ID 时，请查看 `Adjusted results` 中对应版本的 CSV、`id_mapping.csv`、`filter_report.csv` 和 `quality_report.json`。

## 启动可视化工具

后端：

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

可视化后端当前扫描并读取：

```text
Visualization/Final Data/<folderName>/
```

页面数据集版本显示为：

```text
<folderName> / final
```

如果页面显示 `Final Data 为空` 或 `Final Data 缺少必要 CSV`，请先运行转换，并确认 `Final Data/<folderName>/` 下存在三张标准 CSV。

页面上的 `重新扫描` 按钮会调用后端 `/api/scan`，等价于执行转换；勾选 `覆盖已有` 时会以 `force=true` 重新转换。

## 背景图与车道

选择数据集后，后端会优先读取：

```text
Visualization/Initial results/<folderName>/background_*.jpg
```

如果没有背景图，则使用：

```text
Visualization/Initial results/<folderName>/first_frame_*.jpg
```

标准 CSV 中的 world/meter 坐标会由后端结合 pkl 里的像素四点框和 world 四点框估计 `world -> pixel` 映射，再返回给前端绘制，从而与背景图对齐。

`显示车道` 开关会尝试读取 `Visualization/Initial results/<folderName>/` 下的 json，并识别其中的 `road`、`lane_*`、`laneline_*`、`drivingline*` 形状。如果没有对应 road config，前端会提示当前数据集没有可绘制的车道几何；目标详情和悬停提示仍会显示当前帧 `lane_id`。
