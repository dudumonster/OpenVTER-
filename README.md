OpenVTER
===============

## 项目说明

OpenVTER 是一个基于旋转框的车辆轨迹提取框架。当前仓库的主流程仍然是视频稳定、目标检测、跟踪与轨迹输出。

这次仓库已经补上了一套更适合以下场景的工程化入口：

- Windows / Mac 本地开发
- Git 或 rsync 同步代码
- Linux 服务器运行
- 使用不同配置文件切换本地调试与服务器正式任务

标准入口现在统一为：

```bash
python src/train.py --config configs/local_windows.yaml
python src/train.py --config configs/local_mac.yaml
python src/train.py --config configs/server.yaml
```

说明：

- 这里的 `train.py` 是统一任务启动器，当前标准化的是 OpenVTER 现有的 `video_inference` 主流程。
- 它会读取 YAML 配置、解析路径、创建输出目录，并生成一份运行时 JSON 配置再交给原有推理逻辑执行。
- 原始入口 `video_inference_main.py` 仍然保留，并兼容 `--config` 参数。

## 目录约定

新增或整理后的关键目录：

- `configs/`
  - `local_windows.yaml`
  - `local_mac.yaml`
  - `server.yaml`
- `scripts/`
  - `run_local_windows.ps1`
  - `run_local_mac.sh`
  - `run_server.sh`
  - `submit_job.sh`
- `src/`
  - `train.py`

推荐约定：

- 输入数据放在 `data/`、`data1/` 或服务器数据目录
- 运行输出放在 `outputs/`
- 日志放在 `logs/`
- 模型权重放在 `checkpoints/`

## 本地开发流程

### 1. 同步代码

```bash
git pull
```

### 2. 修改代码

直接在本地 IDE 中开发即可。建议只改源码和配置，不要把运行结果提交到 Git。

### 3. 提交代码

```bash
git add .
git commit -m "your message"
git push
```

### 4. 本地运行

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local_windows.ps1
```

Mac:

```bash
bash scripts/run_local_mac.sh
```

也可以直接运行：

```bash
python src/train.py --config configs/local_windows.yaml
python3 src/train.py --config configs/local_mac.yaml
```

## 服务器运行流程

### 1. 登录服务器

```bash
ssh your_user@your_server
```

### 2. 进入项目目录

```bash
cd /path/to/OpenVTER
```

### 3. 同步代码

可以使用 Git：

```bash
git pull
```

也可以使用 rsync：

```bash
rsync -av --delete ./ your_user@your_server:/path/to/OpenVTER/
```

### 4. 激活环境

示例：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvter
```

或者：

```bash
source ~/venvs/openvter/bin/activate
```

### 5. 服务器直接测试运行

```bash
bash scripts/run_server.sh
```

或者：

```bash
python3 src/train.py --config configs/server.yaml
```

### 6. 提交任务

如果服务器使用 Slurm：

```bash
sbatch scripts/submit_job.sh
```

如果不是 Slurm，请按你的调度系统修改 `scripts/submit_job.sh` 中的头部参数和提交流程。

## 日志查看

直接测试运行时，可以看标准输出。

Slurm 任务日志示例：

```bash
tail -f logs/slurm-123456.out
tail -f logs/slurm-123456.err
```

如果你把业务日志写入 `logs/` 下，也可以直接查看：

```bash
tail -f logs/server/your_log_file.out
```

## 结果下载

从服务器下载输出结果：

```bash
scp -r your_user@your_server:/path/to/OpenVTER/outputs ./outputs_from_server
scp -r your_user@your_server:/path/to/OpenVTER/logs ./logs_from_server
scp -r your_user@your_server:/path/to/OpenVTER/checkpoints ./checkpoints_from_server
```

或者使用 rsync：

```bash
rsync -av your_user@your_server:/path/to/OpenVTER/outputs/ ./outputs_from_server/
rsync -av your_user@your_server:/path/to/OpenVTER/logs/ ./logs_from_server/
rsync -av your_user@your_server:/path/to/OpenVTER/checkpoints/ ./checkpoints_from_server/
```

## 配置说明

当前提供了三份环境配置：

- `configs/local_windows.yaml`
  - 用于 Windows 本地小规模调试
  - 默认 `cpu`
  - 较小 `batch_size`
- `configs/local_mac.yaml`
  - 用于 Mac 本地小规模调试
  - 默认 `cpu`
  - 较小 `batch_size`
- `configs/server.yaml`
  - 用于 Linux 服务器
  - 默认 `cuda:0`
  - 更大的 `batch_size`
  - 使用服务器路径

这些配置至少包含：

- `data_dir`
- `output_dir`
- `log_dir`
- `checkpoint_dir`
- `batch_size`
- `num_workers`
- `device`
- `epochs`
- `learning_rate`

同时也保留了 OpenVTER 实际运行需要的参数，例如：

- `video_folder`
- `road_config`
- `pipeline`
- `detection`
- `tracking`
- `stabilize_file`

## 依赖安装

基础依赖在 `requirements.txt` 中。

建议先安装基础包：

```bash
pip install -r requirements.txt
```

然后根据你的服务器环境单独安装：

- `torch`
- `torchvision`
- `mmcv`
- `mmdet`
- `mmrotate`

这些包和 CUDA / Python 版本强相关，不建议在仓库里写死一个不确定可用的版本组合。

## 兼容性说明

这次改造重点保证了以下几点：

- 不再依赖当前工作目录的偶然位置
- 本地和服务器通过不同 YAML 配置切换
- 运行时路径统一从配置读取
- 输出目录会自动创建
- Windows / Mac / Linux 统一走 `src/train.py`

同时保留了原有核心推理逻辑，避免大改模型和算法实现。
