import pickle
import pandas as pd
import os

def pkl_to_excel(file_path, output_folder):
    """
    将 .pkl 文件中的检测结果转换为 Excel 文件并保存。
    
    参数:
        file_path (str): 输入的 .pkl 文件路径
        output_folder (str): 输出文件夹路径，Excel 文件将保存在该文件夹下
    
    返回:
        None
    """
    try:
        # 加载 .pkl 文件
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # 检查数据结构
        if isinstance(data, list):
            print(f"该 pkl 文件包含 {len(data)} 帧的检测结果。")
        else:
            print("警告: 该 pkl 文件结构不符合预期，数据可能不是列表。")
            return
        
        # 准备数据：将数据转为 DataFrame
        rows = []
        for frame_data in data:
            frame_index = frame_data.get('frame', '未知帧')
            bboxes = frame_data.get('bboxes', [])
            scores = frame_data.get('scores', [])
            categories = frame_data.get('categories', [])
            
            # 将每帧的检测框、置信度和类别存储为一行
            for i, bbox in enumerate(bboxes):
                row = {
                    'Frame': frame_index,
                    'BBox': bbox,
                    'Score': scores[i] if i < len(scores) else 'N/A',
                    'Category': categories[i] if i < len(categories) else 'N/A'
                }
                rows.append(row)
        
        # 转换为 DataFrame
        df = pd.DataFrame(rows)
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 保存为 Excel 文件
        output_file = os.path.join(output_folder, 'detection_results.xlsx')
        df.to_excel(output_file, index=False)
        
        print(f"Excel 文件已保存至 {output_file}")
    
    except Exception as e:
        print(f"加载 .pkl 文件时出错：{e}")

# 使用示例
file_path = '路径_to_your_det_bbox_result_cropped_1min.pkl'  # 替换为你的 .pkl 文件路径
output_folder = '路径_to_your_using_folder/new_folder'  # 替换为你的输出文件夹路径
pkl_to_excel(file_path, output_folder)
