# OpenVTER 项目环境说明

当前项目已创建独立 conda 环境：

```text
OpenVTER
```

环境位置：

```text
D:\Software\anaconda\envs\OpenVTER
```

## 为什么使用 Python 3.7

当前本机 Anaconda 是 32 位版本：

```text
platform: win-32
Python: 32 bit
```

直接创建较新的 Python 3.8/3.9 环境时，conda 解析依赖会出现内存不足。因此当前 `OpenVTER` 环境使用 Python 3.7，并安装与当前机器兼容、且已经验证可运行转换模块的依赖版本。

## 已安装核心依赖

```text
python 3.7.13
numpy 1.15.1
scipy 1.1.0
pandas 0.23.4
Pillow 5.2.0
PyYAML 3.13
```

这些依赖已经验证可以：

```text
1. 导入 Visualization/app/converter.py
2. 读取 det_bbox_result_*.pkl
3. 使用 scipy 的 Savitzky-Golay 平滑和 PCHIP 插值函数
```

## 使用方法

每次做当前项目相关操作前，先打开 Anaconda Prompt 或 PowerShell，然后执行：

```powershell
conda activate OpenVTER
cd /d D:\OpenVTER-
```

如果 PowerShell 中 `conda activate` 不可用，可以先运行：

```powershell
D:\Software\anaconda\Scripts\activate OpenVTER
```

## 数据集转换

转换全部数据集：

```powershell
python "Visualization\app\converter.py" --force
```

只转换某个数据集：

```powershell
1 python "Visualization\app\converter.py" --datasets cao_qiao_001 --force
python "Visualization\app\converter.py" --datasets qian_qi_neng_yuan_020 --force
2 python "Visualization\app\converter.py" --datasets cao_qiao_001 --force > "logs\cao_qiao_001.log" 2>&1

2 python "Visualization\app\converter.py" --datasets qian_qi_neng_yuan_020 --force > "logs\qian_qi_neng_yuan_020.log" 2>&1

3 python "Visualization\app\converter.py" --datasets yin_hai_1_016 --force > "logs\yin_hai_1_016.log" 2>&1

4 python "Visualization\app\converter.py" --datasets ban_xian_shan_008 --force > "logs\ban_xian_shan_008.log" 2>&1


```

检查 pkl 结构：

```powershell
python "Visualization\app\converter.py" --inspect "Visualization\Initial results\cao_qiao_001\det_bbox_result_cao_qiao_001.pkl"
```

## 启动可视化工具

```powershell
python "Visualization\app\server.py" --host 127.0.0.1 --port 8000
```

然后浏览器打开：

```text
http://127.0.0.1:8000
```

## 复现环境

项目根目录提供了环境配置文件：

```text
environment_OpenVTER.yml
```

如果以后需要重新创建环境，可以先删除旧环境：

```powershell
conda remove -n OpenVTER --all
```

再重新创建：

```powershell
conda env create -f environment_OpenVTER.yml
```

注意：当前配置是为本机 32 位 Anaconda 准备的。如果后续更换为 64 位 Miniconda/Anaconda，可以升级到更现代的 Python 和依赖版本。
