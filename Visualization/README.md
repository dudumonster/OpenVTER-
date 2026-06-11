# OpenVTER Visualization

本目录用于把 OpenVTER 的 `det_bbox_result_*.pkl` 轨迹结果转换为标准 CSV，并通过本地网页工具查看轨迹、目标框、类别、车道和背景图对齐效果。

## 文档入口

专题文档只保留两个：

```text
VISUALIZATION_COMMANDS.md       常用运行命令、环境、转换、服务启动和轨迹检查脚本
VISUALIZATION_PROJECT_FLOW.md   项目流程、数据结构、输出目录、字段和转换逻辑
```

`README.md` 只作为入口页。

## 快速开始

在项目根目录执行：

```powershell
Set-Location "D:\OpenVTER-"
conda activate OpenVTER
```

转换全部数据集：

```powershell
python "Visualization\app\converter.py" --force
```

启动可视化服务：

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 目录概览

```text
Visualization/
├─ Initial results/       原始输入
├─ Adjusted results/      完整中间结果和质量追溯文件
├─ Final Data/            正式输出 CSV
├─ Trajs Check/           轨迹运动学检查脚本
├─ logs/                  日志
└─ app/                   转换器、服务端和前端资源
```

更多命令见 `VISUALIZATION_COMMANDS.md`；完整流程见 `VISUALIZATION_PROJECT_FLOW.md`。
