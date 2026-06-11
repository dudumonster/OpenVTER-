# Visualization 指令集合

本文只收集 `D:\OpenVTER-\Visualization` 相关的常用命令。默认在项目根目录执行：

```powershell
Set-Location "D:\OpenVTER-"
```

## 1. 启用环境

PowerShell:

```powershell
conda activate OpenVTER
```

如果当前终端无法直接使用 `conda activate`:

```powershell
& "D:\Software\anaconda\Scripts\activate" OpenVTER
```

cmd 或 Anaconda Prompt:

```cmd
conda activate OpenVTER
cd /d D:\OpenVTER-
```

当前项目推荐使用的环境文件：

```text
environment_OpenVTER.yml
```

当前本机环境位置：

```text
D:\Software\anaconda\envs\OpenVTER
```

## 2. 数据转换

转换全部数据集：

```powershell
python "Visualization\app\converter.py" --force
```

转换单个数据集：

```powershell
python "Visualization\app\converter.py" --datasets cao_qiao_001 --force
```

转换多个指定数据集：

```powershell
python "Visualization\app\converter.py" --datasets cao_qiao_001 qian_qi_neng_yuan_020 yin_hai_1_016 ban_xian_shan_008 --force
```

指定输入目录、中间输出目录和正式输出目录：

```powershell
python "Visualization\app\converter.py" --source-root "Visualization\Initial results" --output-root "Visualization\Adjusted results" --final-output-root "Visualization\Final Data" --force
```

检查某个 pkl 的结构：

```powershell
python "Visualization\app\converter.py" --inspect "Visualization\Initial results\cao_qiao_001\det_bbox_result_cao_qiao_001.pkl"
```

把控制台输出同时保存到日志：

```powershell
python "Visualization\app\converter.py" --datasets cao_qiao_001 --force *> "Visualization\logs\cao_qiao_001_console.log"
```

转换日志位置：

```text
Visualization\logs\dataset_conversion.log
Visualization\Adjusted results\<folderName>\<version>\conversion_log.txt
```

## 3. 启动可视化服务

启动后端：

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

如果 8000 端口被占用：

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8001
```

对应地址：

```text
http://127.0.0.1:8001
```

可视化服务日志：

```text
Visualization\logs\visualization_server.log
```

## 4. 轨迹运动学检查图

进入脚本目录：

```cmd
conda activate OpenVTER
cd /d "D:\OpenVTER-\Visualization\Trajs Check"
```

查看某个数据集的所有 track。每关闭一个图窗，自动进入下一个 track：

```cmd
python trajectory_check_visualizer.py --data_root "D:\OpenVTER-\Visualization\Final Data" --folder cao_qiao_001
```

只查看指定 track：

```cmd
python trajectory_check_visualizer.py --data_root "D:\OpenVTER-\Visualization\Final Data" --folder cao_qiao_001 --track_id 25
```

查看数据集整体统计图：

```cmd
python trajectory_check_visualizer.py --data_root "D:\OpenVTER-\Visualization\Final Data" --folder cao_qiao_001 --summary
```

不指定 `--folder` 时，程序会列出 `data_root` 下可用的数据集：

```cmd
python trajectory_check_visualizer.py --data_root "D:\OpenVTER-\Visualization\Final Data"
```

检查图使用 `matplotlib` 窗口交互显示；当前脚本不保存图片，也不保存 CSV。

## 5. 重新创建环境

删除旧环境：

```powershell
conda remove -n OpenVTER --all
```

按项目环境文件重建：

```powershell
conda env create -f environment_OpenVTER.yml
```

当前环境文件主要覆盖转换与可视化所需能力，包括 `python 3.7`、`numpy`、`scipy`、`pandas`、`Pillow` 和 `PyYAML`。
