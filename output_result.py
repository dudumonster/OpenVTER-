import pickle
import pandas as pd

file_path = "data1/UAV_Videos/20220303_5_E_300/output/20220303_5_E_300_1_Num_3/det_bbox_result_20220303_5_E_300_1.pkl"

# 读取 pkl 文件
with open(file_path, "rb") as f:
    data = pickle.load(f)
  
records = []
for traj_id, _, arr in data['traj_info']:
    for frame_id, row in enumerate(arr):
        record = {"traj_id": traj_id, "frame_id": frame_id}
        for i, val in enumerate(row):
            record[f"col_{i}"] = val
        records.append(record)

df = pd.DataFrame(records)

# 导出为 Excel 和 CSV
df.to_csv("trajectories.csv", index=False)
df.to_excel("trajectories.xlsx", index=False)

print("✅ 导出完成：trajectories.csv / trajectories.xlsx")
