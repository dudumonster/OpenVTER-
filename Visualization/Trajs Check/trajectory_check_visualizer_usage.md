# 轨迹运动学可视化检查脚本使用说明

进入脚本目录：

```cmd
conda activate OpenVTER
cd /d "D:\OpenVTER-\Visualization\Trajs Check"
```

如果 `conda activate` 在当前终端不可用，可以使用：

```cmd
"D:\Software\anaconda\Scripts\activate" OpenVTER
cd /d "D:\OpenVTER-\Visualization\Trajs Check"
```

逐个交互式查看某个数据集的所有 track：
逐个 track 弹出检查图。每关闭一个窗口，自动进入下一个 trackId。每个窗口都是单个 track 的 4x3 联图，包含轨迹、速度着色、加速度着色、heading、速度/加速度分量、一致性误差、跳变量、异常计数和文本统计。
```cmd
python trajectory_check_visualizer.py --data_root "D:\OpenVTER-\Visualization\Final Data" --folder cao_qiao_001
```

只查看某个数据集中的指定 track：
图表内容和上面单个 track 的联图一样，但只显示 trackId=25，关闭窗口后程序直接结束。
```cmd
python trajectory_check_visualizer.py --data_root "D:\OpenVTER-\Visualization\Final Data" --folder cao_qiao_001 --track_id 25
```

查看某个数据集的整体统计图：
输出的是整个 folder 的全局统计图，不逐个显示轨迹。它主要是速度、加速度、lon/lat 分量、heading 跳变量的分布直方图，speed/acc 一致性散点图，不同 class 的箱线图，以及全局异常数量统计。
```cmd
python trajectory_check_visualizer.py --data_root "D:\OpenVTER-\Visualization\Final Data" --folder cao_qiao_001 --summary
```

不指定 `--folder` 时，程序会列出 `data_root` 下可用的子文件夹：

```cmd
python trajectory_check_visualizer.py --data_root "D:\OpenVTER-\Visualization\Final Data"
```

当前版本只通过 `matplotlib` 窗口交互式查看结果，不保存图片，不保存 CSV。
