# 从data.npz和info.pkl中提取数据并生成PoseCNN训练所需的数据格式
import numpy as np
import cv2
import pickle

data = np.load('data.npz')

rgb = data["rgb"]
idx = data["id"]

info = pickle.load(open("info.pkl", "rb"))

# 存储姿态和边界框
json_data = {
}
json_data_info = {
}

import os

if not os.path.exists("export"):
    os.mkdir("export")

import shutil
shutil.rmtree("export/rgb")
os.mkdir("export/rgb")
shutil.rmtree("export/mask_visib")
os.mkdir("export/mask_visib")

for i in range(len(rgb)):
    img_json_data = []      # 当前图像的物体姿态数据
    img_json_data_info = [] # 当前图像的边界框数据
    
    # 保存RGB图像（转换为BGR格式）
    bgr_image = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"export/rgb/{i+100000:06}.png", bgr_image)
    
    # 处理当前图像中的每个物体
    ids = np.unique(idx[i])
    for index, j in enumerate(ids):
        # 跳过ID为0的背景
        if j == 0:
            continue
            
        # 获取当前物体的姿态信息
        img_obj_info = info[i][j]
        RT = img_obj_info["RT"]
        RT = np.linalg.inv(RT)
        
        # 姿态信息
        img_obj_json_data = {
            "cam_R_m2c": RT[:3, :3].flatten().tolist(),
            "cam_t_m2c": RT[:3, 3].flatten().tolist(),
            "obj_id": int(j)# 物体ID
        }
        img_json_data.append(img_obj_json_data)

        # 生成mask
        mask = np.zeros_like(idx[i])
        mask[idx[i] == j] = 255

        # 计算物体边界框
        x_min = np.min(np.where(mask > 0)[1])
        x_max = np.max(np.where(mask > 0)[1])
        y_min = np.min(np.where(mask > 0)[0])
        y_max = np.max(np.where(mask > 0)[0])

        # 边界框信息
        img_obj_json_data_info = {"bbox_visib": [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)]}
        img_json_data_info.append(img_obj_json_data_info)

        # 物体mask
        cv2.imwrite(f"export/mask_visib/{i + 100000:06}_{index-1:06}.png", mask)
    
    json_data[i + 100000] = img_json_data
    json_data_info[i + 100000] = img_json_data_info

import json
# 姿态信息
with open("export/train_gt.json", "w") as f:
    json.dump(json_data, f)

# 边界框信息
with open("export/train_gt_info.json", "w") as f:
    json.dump(json_data_info, f)
