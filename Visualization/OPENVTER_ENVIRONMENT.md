# OpenVTER 项目环境说明

当前项目使用独立 conda 环境：

```text
OpenVTER
```

环境位置：

```text
D:\Software\anaconda\envs\OpenVTER
```

## 当前环境版本

项目根目录提供了环境文件：

```text
environment_OpenVTER.yml
```

当前环境文件内容对应：

```text
python 3.7
numpy 1.15.1
scipy 1.1.0
pandas 0.23.4
Pillow 5.2.0
PyYAML 3.13
```

这些依赖已经覆盖当前转换与可视化所需功能：

```text
1. 导入 Visualization/app/converter.py
2. 读取 det_bbox_result_*.pkl
3. 使用 numpy 处理轨迹矩阵
4. 使用 scipy 的 Savitzky-Golay 平滑和 PCHIP 插值
5. 使用 Pillow 读取背景图尺寸
```

注意：`Visualization/app/requirements.txt` 是较新 Python 环境可用的 pip 依赖范围；当前这台机器推荐优先使用 `environment_OpenVTER.yml` 中已经验证过的 conda 环境。

## 启用环境

PowerShell：

```powershell
conda activate OpenVTER
Set-Location "D:\OpenVTER-"
```

如果 PowerShell 中 `conda activate` 不可用，可以运行：

```powershell
& "D:\Software\anaconda\Scripts\activate" OpenVTER
Set-Location "D:\OpenVTER-"
```

Anaconda Prompt 或 cmd：

```cmd
conda activate OpenVTER
cd /d D:\OpenVTER-
```

## 数据集转换

转换全部数据集：

```powershell
python "Visualization\app\converter.py" --force
```

只转换某一个数据集：

```powershell
python "Visualization\app\converter.py" --datasets cao_qiao_001 --force
```

只转换多个指定数据集：

```powershell
python "Visualization\app\converter.py" --datasets cao_qiao_001 qian_qi_neng_yuan_020 yin_hai_1_016 ban_xian_shan_008 --force
```

把控制台输出同时保存到日志文件：

```powershell
python "Visualization\app\converter.py" --datasets cao_qiao_001 --force *> "Visualization\logs\cao_qiao_001_console.log"
```

检查 pkl 结构：

```powershell
python "Visualization\app\converter.py" --inspect "Visualization\Initial results\cao_qiao_001\det_bbox_result_cao_qiao_001.pkl"
```

转换程序自己的总日志写入：

```text
Visualization/logs/dataset_conversion.log
```

转换会同时生成：

```text
Visualization/Adjusted results/<folderName>/full/
Visualization/Adjusted results/<folderName>/moving_filtered/
Visualization/Final Data/<folderName>/
```

## 启动可视化工具

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8000
```

然后在浏览器打开：

```text
http://127.0.0.1:8000
```

可视化服务日志写入：

```text
Visualization/logs/visualization_server.log
```

如果 8000 端口被占用，可以换一个端口：

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8001
```

对应浏览器地址：

```text
http://127.0.0.1:8001
```

## 复现环境

如果需要重新创建环境，可以先删除旧环境：

```powershell
conda remove -n OpenVTER --all
```

再重新创建：

```powershell
conda env create -f environment_OpenVTER.yml
```

当前环境文件是为本机现有 Anaconda 配置准备的。如果后续更换为 64 位 Miniconda/Anaconda，可以再评估是否升级到更新的 Python、numpy、scipy 和 Pillow 版本。
