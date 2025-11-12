import pickle

def load_and_inspect_pkl(file_path):
    """
    加载并检查 .pkl 文件的内容。

    参数:
        file_path (str): .pkl 文件的路径

    返回:
        data: 加载的文件内容
    """
    # 加载 pkl 文件
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # 打印数据，查看内容
    print(data)
    return data

# 使用示例
file_path = 'data1/UAV_Videos/cropped_1min/output/cropped_1min/cropped_1min.pkl'  # 替换为你的文件路径
loaded_data = load_and_inspect_pkl(file_path)
