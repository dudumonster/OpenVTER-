# OpenVTER 专利流程图候选稿

以下流程图基于项目真实实现链路整理，适合作为专利摘要附图或说明书附图初稿。

依据的核心实现包括：
- 视频主流程：`video_inference/video_process.py`
- 融合配置：`config/demo_config/video_config/20220303_5_E_300_fusion_yolov5.json`
- 道路配置解析：`utils/config.py`
- 轨迹补全与物理约束后处理：`using/track_gap_fill_export.py`

建议附图风格：
- 使用黑白线框
- 每个框内尽量控制在 10-22 字
- 摘要附图优先保留主链路，不展开参数和公式

## 方案A：摘要附图紧凑纵向版

适用场景：摘要页右侧，信息密度适中，最接近常见专利摘要附图。

```mermaid
flowchart TD
    A["S1 获取无人机俯视视频与道路配置数据"]
    B["S2 视频稳像、检测掩膜约束与重叠分块处理"]
    C["S3 旋转框车辆检测与多类目标检测并行推理"]
    D["S4 类别映射与旋转框统一表达"]
    E["S5 全局坐标回投与旋转非极大值抑制融合"]
    F["S6 按类别分组的多跟踪器关联与统一编号管理"]
    G["S7 像素-世界坐标映射、车道归属、车辆长度重分类与类别平滑"]
    H["S8 缺帧分级补全、运动学平滑、静止门控与物理一致性约束"]
    I["S9 输出结构化轨迹结果与可视化结果"]

    A --> B --> C --> D --> E --> F --> G --> H --> I

    classDef patent fill:#ffffff,stroke:#666666,color:#000000,stroke-width:1px;
    class A,B,C,D,E,F,G,H,I patent;
    linkStyle default stroke:#555555,stroke-width:1.2px;
```

## 方案B：摘要附图分支融合版

适用场景：强调“多模型融合”这一创新点，更适合突出你的发明标题。

```mermaid
flowchart TD
    A1["无人机俯视视频"]
    A2["道路配置数据"]
    B["稳像、掩膜裁剪与分块处理"]
    C1["旋转框车辆检测模型"]
    C2["多类交通参与者检测模型"]
    D["类别映射与统一旋转框表达"]
    E["全局旋转非极大值抑制"]
    F1["机动车跟踪器组"]
    F2["非机动车与行人跟踪器组"]
    G["统一ID管理与连续轨迹输出"]
    H["世界坐标映射、车道归属与车辆细分类"]
    I["轨迹类别时序平滑、缺帧补全与物理约束"]
    J["结构化轨迹文件与可视化结果"]

    A1 --> B
    A2 --> B
    B --> C1
    B --> C2
    C1 --> D
    C2 --> D
    D --> E
    E --> F1
    E --> F2
    F1 --> G
    F2 --> G
    G --> H --> I --> J

    classDef patent fill:#ffffff,stroke:#666666,color:#000000,stroke-width:1px;
    class A1,A2,B,C1,C2,D,E,F1,F2,G,H,I,J patent;
    linkStyle default stroke:#555555,stroke-width:1.2px;
```

## 方案C：方法总流程步骤版

适用场景：作为说明书中的“图1 方法总体流程图”，步骤编号更强，便于后续扩写权利要求。

```mermaid
flowchart TD
    A([开始])
    B["步骤1 读取无人机视频、稳定区域、检测区域、车道区域及像素-世界映射参数"]
    C["步骤2 对视频帧执行稳像处理，并对检测区域进行掩膜约束"]
    D["步骤3 将当前帧切分为重叠子图像块"]
    E["步骤4 对各子图像块分别执行旋转框检测和多类目标检测"]
    F["步骤5 将多模型检测结果统一映射为旋转框表达，并执行全局融合抑制"]
    G["步骤6 依据类别分组将融合结果输入多个跟踪器，得到统一编号轨迹"]
    H["步骤7 将轨迹映射至世界坐标，进行车道判定、车辆长度重分类与类别平滑"]
    I["步骤8 对轨迹进行缺帧补全、速度加速度计算、静止门控和物理约束校验"]
    J["步骤9 输出结构化轨迹结果、统计结果及可视化结果"]
    K([结束])

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K

    classDef patent fill:#ffffff,stroke:#666666,color:#000000,stroke-width:1px;
    class B,C,D,E,F,G,H,I,J patent;
    classDef terminal fill:#ffffff,stroke:#666666,color:#000000,stroke-width:1px;
    class A,K terminal;
    linkStyle default stroke:#555555,stroke-width:1.2px;
```

## 推荐意见

- 如果你现在是为了补摘要右侧的小流程图，优先选“方案A”。
- 如果你想强调本发明区别于传统“单检测器+单跟踪器”的创新点，优先选“方案B”。
- 如果你后面还要继续写说明书附图说明和实施方式，建议把“方案C”作为图1的基础版本。
