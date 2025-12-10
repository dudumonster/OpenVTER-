import cv2

# 输入视频路径
input_video_path = 'data1/UAV_Videos/20220303_5_E_300/track/simpleround_road.mp4'
# 输出视频路径
output_video_path = 'data1/UAV_Videos/20220303_5_E_300/track/test.mp4'

# 设置裁剪时长（秒），可以是60（1分钟）或20（20秒）
duration = 20  # 可以修改为20来裁剪20秒的视频

# 打开视频文件
cap = cv2.VideoCapture(input_video_path)

# 获取视频的基本信息
fps = cap.get(cv2.CAP_PROP_FPS)  # 视频的帧率
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频的总帧数
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频的宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频的高度

# 计算视频总时长（秒）
total_duration = frame_count / fps
print(f"原始视频总时长: {total_duration:.2f}秒（约{total_duration/60:.2f}分钟）")

# 计算裁剪的起始帧和结束帧
start_frame = 0  # 从第0帧开始（视频开始）
end_frame = int(fps * duration) - 1  # 根据duration计算结束帧

# 确保结束帧不超过视频总帧数
if end_frame >= frame_count:
    end_frame = frame_count - 1
    print(f"警告: 视频长度不足{duration}秒，将裁剪到视频结尾（第{end_frame}帧）")

print(f"将从第{start_frame}帧裁剪到第{end_frame}帧，裁剪后的视频长度约为{(end_frame - start_frame + 1) / fps:.2f}秒")

# 打开视频写入对象（保存裁剪后的部分）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 从起始帧开始读取视频，直到结束帧
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 设置视频读取的起始帧

# 读取视频帧并写入输出视频
frame_num = start_frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num > end_frame:
        break
    
    # 将帧写入新视频文件
    out.write(frame)
    
    frame_num += 1

# 释放视频资源
cap.release()
out.release()

print(f"视频裁剪完成，已保存为{duration}秒长度的视频: {output_video_path}")
