# OpenVTER 轨迹可视化与 pkl 转 CSV

该目录用于把 OpenVTER 每个地点的 pkl 结果转换成标准化 CSV，并通过本地后端 + Canvas 前端查看轨迹。

## 目录结构

```text
Visualization/
├── Initial results/
│   └── cao_qiao_001/
│       ├── background_cao_qiao_001.jpg
│       ├── cao_qiao_001_stab.pkl
│       ├── det_bbox_result_cao_qiao_001.pkl
│       ├── first_frame_cao_qiao_001.jpg
│       └── tracking_output_stab_det_cao_qiao_001.mp4
├── Adjusted results/
│   └── cao_qiao_001/
│       ├── full/
│       │   ├── tracks.csv
│       │   ├── objects.csv
│       │   ├── frames.csv
│       │   ├── metadata.json
│       │   └── background.jpg
│       └── moving_filtered/
│           ├── tracks.csv
│           ├── objects.csv
│           ├── frames.csv
│           ├── metadata.json
│           └── background.jpg
├── logs/
└── app/
```

## 放入新数据集

把每个地点的完整结果文件夹拖入：

```text
Visualization/Initial results/
```

例如：

```text
Visualization/Initial results/cao_qiao_001/
Visualization/Initial results/cao_qiao_002/
```

每个子文件夹会被当作一个地点数据集。文件名不会写死为 `cao_qiao_001`，转换器会自动识别 `det_bbox_result_*.pkl`、`*_stab.pkl`、`background_*.jpg` 或 `first_frame_*.jpg`。

## 执行转换

在项目根目录运行：

```powershell
python "Visualization\app\converter.py"
```

覆盖已有转换结果：

```powershell
python "Visualization\app\converter.py" --force
```

只检查某个 pkl 结构：

```powershell
python "Visualization\app\converter.py" --inspect "Visualization\Initial results\cao_qiao_001\det_bbox_result_cao_qiao_001.pkl"
```

转换日志写入：

```text
Visualization/logs/conversion.log
```

## full 与 moving_filtered

每个地点会输出两个版本：

- `full`：完整版本，保留所有目标，不做静止过滤。
- `moving_filtered`：运动过滤版本，只剔除长期静止的机动车目标，保留明显运动目标。

前端数据集列表会显示：

```text
cao_qiao_001 / full
cao_qiao_001 / moving_filtered
```

用户可自行选择要看的版本。`full` 永远保留完整结果，`moving_filtered` 只是额外过滤版本。

## 静止门控

静止门控参数在 [converter.py](app/converter.py) 顶部：

```python
STATIC_GATE_CONFIG = {
    "min_track_length": 30,
    "max_displacement": 10.0,
    "max_mean_speed": 0.5,
    "static_ratio_threshold": 0.8,
    "per_frame_motion_threshold": 1.0,
}
```

含义：

- `min_track_length`：轨迹长度超过该帧数才参与静止过滤。
- `max_displacement`：起点到终点总位移低于该像素阈值，可能是静止目标。
- `max_mean_speed`：平均速度低于该像素/帧阈值，可能是静止目标。
- `static_ratio_threshold`：低速帧比例高于该阈值，可能是静止目标。
- `per_frame_motion_threshold`：单步速度低于该值时，视为静止帧。

默认只对机动车类别执行静止过滤：

```text
car, truck, bus, freight_car, van
```

行人、非机动车默认不过滤，避免误删慢速移动目标。

`objects.csv` 会记录：

```text
displacement, path_length, mean_speed, max_speed, static_ratio, is_static, filter_reason
```

`metadata.json` 会记录版本名、静止门控参数、过滤目标数量和被过滤目标 ID。

## pkl 字段映射

真实样本 `det_bbox_result_cao_qiao_001.pkl` 已解析确认：

- 顶层字段：`video_info`, `output_info`, `traj_info`, `process_time`, `raw_det`
- `traj_info` 每帧为 `(frame_id, output_frame, ndarray)` 或 `(frame_id, output_frame, ndarray, frame_time)`
- 样本数组为 `N x 20`

列映射：

| 列 | 输出字段 | 含义 |
| --- | --- | --- |
| 0-7 | `q1_x,q1_y,...,q4_x,q4_y` | 像素四点旋转 bbox |
| 8 | `confidence` | 置信度 |
| 9 | `category_id/class_name` | 类别 |
| 10 | `object_id` | 跟踪目标 ID |
| 11-18 | `world_q1_x,...,world_q4_y` | 世界坐标四点 |
| 19 | `lane_id` | 车道 ID |

`tracks.csv` 核心字段：

```text
dataset_id, frame_id, object_id, class_name, confidence,
x1, y1, x2, y2, cx, cy, width, height
```

扩展字段保留：

```text
category_id, output_frame, timestamp, angle_deg,
q1_x...q4_y, world_q1_x...world_q4_y, lane_id, source_row_index
```

前端有四点旋转框时优先画旋转 bbox；没有旋转框时画水平 bbox；没有 bbox 时退化为中心点。

## 启动前端

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

如果修改了前端文件，浏览器按：

```text
Ctrl + F5
```

强制刷新缓存。

## 前端使用

前端只保留一种可视化方式：检测框可视化。

- 有旋转 bbox 时画旋转框。
- 没有旋转 bbox 时画水平 bbox。
- 可开关 bbox。
- 可开关 object_id 标签。
- 可调整轨迹拖尾长度。
- 可按类别筛选。
- 可播放、暂停、上一帧、下一帧、拖动时间轴、调整速度。

### object_id / frame_id

输入框不会自动跳转，只有点击“确认跳转”才生效。

- 只输入 `object_id`：筛选该目标，从当前帧继续显示或播放。
- 只输入 `frame_id`：跳转到指定帧，并保持当前目标筛选条件。
- 同时输入 `object_id` 和 `frame_id`：筛选该目标，并跳转到指定帧。
- 两个输入框都留空并点击确认：恢复全目标显示。
- 非法输入会在画布顶部状态区提示，例如目标不存在或帧号超出范围。

### 类别筛选

类别筛选区提供：

- `全选`
- `取消全选`
- 单独勾选/取消某个类别

状态会显示：

- `全选`
- `全部隐藏`
- `部分选中 x/y`

### 数据集列表

数据集列表有固定最大高度，超过后可以滚动。上方搜索框可按数据集名称或版本过滤，例如输入：

```text
moving
```

可快速找到 `moving_filtered` 版本。

## 方向箭头

方向箭头从当前目标中心点 `cx, cy` 出发，指向平滑后的运动方向。

前端参数在 [app.js](app/static/app.js) 顶部：

```js
const HEADING_CONFIG = {
  heading_smooth_window: 8,
  min_motion_threshold: 2.0,
  arrow_length_scale: 0.8,
  arrow_min_length: 8,
  arrow_max_length: 40,
};
```

计算逻辑：

1. 对每个 `object_id` 查最近 `heading_smooth_window` 帧内的中心点。
2. 用 `cx(t)-cx(t-N)`、`cy(t)-cy(t-N)` 计算平滑位移方向。
3. 如果位移小于 `min_motion_threshold`，认为当前方向无效。
4. 方向有效时更新该目标的 `last_valid_heading` 缓存。
5. 方向无效时沿用该目标上一次有效方向。
6. 如果目标从未有过有效方向，暂时不显示箭头。

因此低速或静止目标不会因为检测抖动而出现乱转箭头。
