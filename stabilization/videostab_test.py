import os
import cv2
from .VideoStabilization import VideoStabilization

# 视频文件路径
video_file_ls = ["stabilization/20220303_5_E_300_3.MP4"]  # 这里替换成你的测试视频路径
save_folder = "./stabilization"  # 输出的文件夹路径

# 创建输出文件夹（如果没有的话）
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 初始化视频稳定对象
vs = VideoStabilization()

# 你可以传入自己选择的掩膜 mask，这里暂时为 None
mask = None  # 这里可以自定义，或者用默认的掩膜
vs.init_stabilize(mask)

# 选择步骤：
# step = 1: 只计算并保存变换（pkl文件）
# step = 2: 只输出稳定视频
# step = 3: 计算并保存变换，同时输出稳定视频
step = 1  # 我们选择 step=3，表示计算变换并输出稳定视频

# 调用 stabilize_video 方法开始稳定处理
vs.stabilize_video(video_file_ls, save_folder, step=step, output_video=True, video_output_fps=30)
