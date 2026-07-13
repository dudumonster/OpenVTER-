# 数据集下载清单

本清单记录当前已经建立的本地目录、下载方式和自动化状态。

## 本地目录

```text
dataset_construction/data_sources/
  dronevehicle/
  uav_obb/
  dota/
  vsai/
  eagle/
  visdrone/
```

每个数据集目录下均预留：

```text
docs/       # 官方页面、README、API 元数据
downloads/  # 原始压缩包
raw/        # 解压后的原始数据
samples/    # 后续抽样预览
```

## 可直接脚本下载

运行方式：

```bash
bash dataset_construction/scripts/download_datasets.sh uav-obb
bash dataset_construction/scripts/download_datasets.sh visdrone
bash dataset_construction/scripts/download_datasets.sh vsai
```

或下载当前建议优先获取的直链数据：

```bash
bash dataset_construction/scripts/download_datasets.sh all-direct
```

`all-direct` 现在只下载 UAV-OBB 和 VisDrone，不再默认下载 VSAI。VSAI 约 13.9GB，且 Supervisely 镜像在当前网络下非常慢，需要单独处理。

下载使用 `curl -C -`，中断后再次运行会断点续传。

## 数据源状态

| 数据集 | 类型 | 当前自动化状态 | 本地目标目录 | 说明 |
| --- | --- | --- | --- | --- |
| UAV-OBB | OBB | 可直链下载 | `data_sources/uav_obb/downloads/` | Mendeley API 最新版本 v4，约 573MB，YOLOv8-OBB 格式 |
| VisDrone-DET | HBB | 已导入 raw 并校验 | `data_sources/visdrone/raw/` | 弱势交通参与者类别最重要来源，包含 bicycle、tricycle、awning-tricycle、motor |
| VSAI | OBB | 已下载并校验 | `data_sources/vsai/downloads/vsai-DatasetNinja.tar` | 约 14GB，Supervisely/Dataset Ninja 格式，包含 small-vehicle / large-vehicle 两类 polygon 标注 |
| DroneVehicle | OBB | 需要官方 BaiduYun/Google Drive 页面下载 | `data_sources/dronevehicle/downloads/` | 机动车 OBB，car/truck/bus/van/freight car；官方 GitHub 提供下载入口 |
| DOTA | OBB | 需要官网 Baidu/Google Drive 下载 | `data_sources/dota/downloads/` | 通用遥感 OBB，交通类主要是 small vehicle / large vehicle |
| EAGLE | OBB/RBB | 需要 DLR 官网下载入口 | `data_sources/eagle/downloads/` | 航空车辆方向检测数据，适合机动车预训练参考 |

## 需要网页登录或网盘下载的数据集入口

### DroneVehicle

官方 GitHub 页面已保存到：

```text
data_sources/dronevehicle/docs/github_page.html
```

百度云入口：

| 分割 | 链接 | 提取码 |
| --- | --- | --- |
| Train | https://pan.baidu.com/s/1ptZCJ1mKYqFnMnsgqEyoGg | `ngar` |
| Validation | https://pan.baidu.com/s/1e6e9mESZecpME4IEdU8t3Q | `jnj6` |
| Test | https://pan.baidu.com/s/1JlXO4jEUQgkR1Vco1hfKhg | `tqwc` |

### DOTA

官网页面已保存到：

```text
data_sources/dota/docs/dota_dataset.html
```

DOTA-v1.0 / v1.5：

| 分割 | Baidu Drive | Google Drive |
| --- | --- | --- |
| Training set | https://pan.baidu.com/s/1kWyRGaz | https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK?usp=sharing |
| Validation set | https://pan.baidu.com/s/1qZCoF72 | https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224?usp=sharing |
| Testing images | https://pan.baidu.com/s/1i6ly9Id | https://drive.google.com/drive/folders/1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl?usp=sharing |

DOTA-v2.0：

| 链接 | 提取码 |
| --- | --- |
| https://pan.baidu.com/s/1Y9rJPEZLCdjyVjMSyhFdDQ | `ck24` |

### EAGLE

DLR 官方页面已保存到：

```text
data_sources/eagle/docs/eagle_dataset.html
```

入口：

```text
https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/eagle
```

## 已尝试的自动下载情况

- VisDrone 已从服务器导入解压后的 `train`、`val`、`test-dev` 到 `data_sources/visdrone/raw/`，并完成图片/标注一一对应检查；`downloads/` 中早先未完成的 `.zip` 可忽略。
- UAV-OBB 直链可访问，但当前速度约 100-250KB/s，已停止等待；已保留部分 `.zip` 文件，后续脚本可断点续传。
- VSAI 已通过自定义 HTTP Range 分片下载器完成下载。最终文件为 `data_sources/vsai/downloads/vsai-DatasetNinja.tar`，`tar -tf` 校验通过，`meta.json` 确认包含 `large-vehicle` 和 `small-vehicle` 两类 polygon 标注。

## VSAI 下载太慢时的处理

如果 Supervisely 直链在本机网络下只有 KB/s 级速度，13.9GB 会需要很久，不建议继续用普通 `curl` 等待。

推荐处理顺序：

1. 暂时跳过 VSAI，优先下载 UAV-OBB、VisDrone 和 DOTA-v1.0。
2. 如需 VSAI，优先尝试 Kaggle 原始格式入口：

```text
https://www.kaggle.com/datasets/dronevision/vsaiv1
```

Kaggle CLI 下载方式：

```bash
pip install kaggle
mkdir -p ~/.kaggle
# 将 Kaggle 账号生成的 kaggle.json 放到 ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d dronevision/vsaiv1 -p dataset_construction/data_sources/vsai/downloads --unzip
```

3. 如果继续使用 Supervisely 直链，可以先安装多线程下载工具，或使用本项目中的 Range 下载脚本：

```bash
brew install aria2
aria2c -x 8 -s 8 -c -d dataset_construction/data_sources/vsai/downloads -o vsai-DatasetNinja.tar 'SUPERVISELY_DOWNLOAD_URL'
```

其中 `SUPERVISELY_DOWNLOAD_URL` 可从 `scripts/download_datasets.sh` 的 `vsai` 分支复制。

```bash
python3 dataset_construction/scripts/ranged_download.py 'SUPERVISELY_DOWNLOAD_URL' dataset_construction/data_sources/vsai/downloads/vsai-DatasetNinja.tar --workers 32 --part-mb 128
```
