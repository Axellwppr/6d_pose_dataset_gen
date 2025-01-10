# 导入必要的库
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from omni.isaac.lab.sensors import CameraCfg, Camera
import random
import math
import torch
from pxr import UsdShade
import omni.physics.tensors.impl.api as physx
from omni.physics.tensors.impl.api import RigidBodyView

device = "cuda:0"

sem_to_index = {}

# 定义相机内参矩阵
intrinsic_matrix = torch.tensor(
    [
        [1066.778, 0.0, 312.9869079589844],  # fx, 0, cx
        [0.0, 1067.487, 241.3108977675438],  # 0, fy, cy
        [0.0, 0.0, 1.0],                     # 0, 0, 1
    ],
    device=device,
)

# 设置图像尺寸
width = 640
height = 480

# 计算图像中心点
center_x = intrinsic_matrix[0, 2]
center_y = intrinsic_matrix[1, 2]

# 计算裁剪区域
width_x = max(center_x, width - center_x)
width_y = max(center_y, height - center_y)

# 计算裁剪边界
start_x = math.floor(center_x - width_x)
end_x = math.ceil(center_x + width_x)
start_y = math.floor(center_y - width_y)
end_y = math.ceil(center_y + width_y)

# 计算裁剪偏移量
crop_x = -start_x
crop_y = -start_y

# 计算新的图像尺寸和更新内参矩阵
new_width = end_x - start_x
new_height = end_y - start_y
intrinsic_matrix[0, 2] = new_width / 2.0
intrinsic_matrix[1, 2] = new_height / 2.0

def design_scene(other_thing: list[str]):
    """设计场景，包括添加地面、光源、物体和网格"""
    global sem_to_index
    
    # 创建地面
    cfg_ground = sim_utils.UsdFileCfg(
        usd_path="./plane.usd",
        scale=(100.0, 100.0, 0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # 创建平行光源
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
        angle=5.0,
    )
    cfg_light_distant.func(
        "/World/lightDistant", cfg_light_distant, translation=(1, 0, 10)
    )

    # 创建环境光
    cfg_dome_light = sim_utils.DomeLightCfg(
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDome", cfg_dome_light, translation=(1, 0, 10))

    # 设置相机
    camera_count = 5
    cameras = []
    for i in range(camera_count):
        camera = Camera(
            CameraCfg(
                prim_path=f"/World/cam/front_cam_{i}",
                update_period=0.1,
                height=new_width,
                width=new_width,
                data_types=["rgb", "instance_segmentation_fast"],
                colorize_instance_segmentation=False,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
            )
        )
        cameras.append(camera)

    prim_utils.create_prim("/World/Objects", "Xform")

    # 导入所有物体
    usd_files = other_thing
    objects = []
    for i, usd_file in enumerate(usd_files):
        usd_id = os.path.basename(os.path.dirname(usd_file))
        sem_to_index[int(usd_id)] = i
        cfg_usd = sim_utils.UsdFileCfg(
            usd_path=usd_file,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02),
            semantic_tags=[("class", usd_id)],
        )
        cfg_rigid = RigidObjectCfg(
            prim_path=f"/World/Objects/UsdFile_{i}",
            spawn=cfg_usd,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(random.uniform(-0.45, 0.45), random.uniform(-0.45, 0.45), 1.3),
                rot=(0.5, 0.0, 0.5, 0.0),
            ),
        )
        rigid_obj: RigidObject = RigidObject(cfg=cfg_rigid)
        objects.append(rigid_obj)

    # import material
    materials = glob.glob("/opt/nvidia/mdl/vMaterials_2/Wood/*.mdl")
    material_path = []
    for index, mat in enumerate(materials):
        material_cfg = sim_utils.MdlFileCfg(mdl_path=mat)
        material = material_cfg.func(f"/World/Looks/woodMaterial_{index}", material_cfg)
        material_path.append(f"/World/Looks/woodMaterial_{index}")

    # 建立桌面
    plane_cfg = sim_utils.UsdFileCfg(
        usd_path="./plane.usd",
        scale=(1.0, 1.0, 0.10),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02),
    )
    plane = plane_cfg.func("/World/Objects/Plane", plane_cfg, translation=(0, 0, 1.0))

    # 建立围墙
    wall1_cfg = sim_utils.CuboidCfg(
        size=(0.01, 1.0, 1.0),
        visible=False,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    wall1_cfg.func("/World/Objects/Wall1", wall1_cfg, translation=(-0.5, 0, 1.0))
    wall2_cfg = sim_utils.CuboidCfg(
        size=(0.01, 1.0, 1.0),
        visible=False,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    wall2_cfg.func("/World/Objects/Wall2", wall2_cfg, translation=(0.5, 0, 1.0))
    wall3_cfg = sim_utils.CuboidCfg(
        size=(1.0, 0.01, 1.0),
        visible=False,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    wall3_cfg.func("/World/Objects/Wall3", wall3_cfg, translation=(0, -0.5, 1.0))
    wall4_cfg = sim_utils.CuboidCfg(
        size=(1.0, 0.01, 1.0),
        visible=False,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    wall4_cfg.func("/World/Objects/Wall4", wall4_cfg, translation=(0, 0.5, 1.0))

    return cameras, objects, material_path


import numpy as np
import cv2


img_count = 0

# 辅助函数：获取四元数和平移向量
def get_quat_trans_from_T(T):
    quat = math_utils.quat_from_matrix(T[:3, :3]).tolist()
    trans = T[:3, 3].tolist()
    return quat, trans

# 辅助函数：构建变换矩阵
def get_T_from_quat_trans(quat, trans):
    quat = torch.as_tensor(quat, device=device).view(4)
    trans = torch.as_tensor(trans, device=device).view(3)
    RT = torch.zeros((4, 4), device=device, dtype=torch.float32)
    RT[:3, :3] = math_utils.matrix_from_quat(quat)
    RT[:3, 3] = trans
    RT[3, 3] = 1
    return RT


def getImg(camera: Camera, objects):
    """获取相机图像和对应的分割信息"""
    global sem_to_index
    
    # 获取RGB图像并裁剪
    rgb_img = camera.data.output["rgb"][0, ..., :3]
    rgb_img = rgb_img.cpu().numpy()[crop_y:crop_y+height, crop_x:crop_x+width]

    # 获取实例分割图像
    id_img = camera.data.output["instance_segmentation_fast"][0, ..., 0].cpu().numpy()[:height,:width]

    obj_info = {}
    id_to_sem = camera.data.info[0]["instance_segmentation_fast"]["idToSemantics"]

    id_data = np.zeros((id_img.shape[0], id_img.shape[1]), dtype=np.uint8)

    cam_pos = camera.data.pos_w[0]
    cam_quat = camera.data.quat_w_ros[0]
    cam_RT = get_T_from_quat_trans(cam_quat, cam_pos)

    for i, sem in id_to_sem.items():
        class_name = sem["class"]
        if class_name in ["BACKGROUND", "UNLABELLED"]:
            continue
        id_data[id_img == i] = int(class_name)
        index = sem_to_index[int(class_name)]
        obj: RigidObject = objects[index]
        obj_pos = obj.data.body_pos_w[0]
        obj_quat = obj.data.body_quat_w[0]
        obj_RT = get_T_from_quat_trans(obj_quat, obj_pos)
        RT_s_c = torch.matmul(torch.linalg.inv(obj_RT), cam_RT) # 世界坐标系到相机坐标系的变换矩阵
        obj_info[int(class_name)] = {"RT": RT_s_c.cpu().numpy()}

    return rgb_img, id_data, obj_info


data = {
    "rgb": [],
    "id": [],
    "info": [],
    "count": 0
}

def saveImg(cameras, objs):
    """保存图像和对应的分割信息"""
    global img_count, data
    for i, camera in enumerate(cameras):
        rgb_img, id_img, obj_info = getImg(camera, objs)
        data["rgb"].append(rgb_img)
        data["id"].append(id_img)
        data["info"].append(obj_info)
        data["count"] += 1
        print("count ", data["count"])


def update_obj(objs):
    """更新物体状态"""
    for obj in objs:
        obj.update(0.02)


def random_light():
    """随机设置光源位置、颜色、亮度，包括线光源和点光源"""
    intensity = random.uniform(1000, 3000)
    prim_utils.set_prim_attribute_value(
        prim_path="/World/lightDistant",
        attribute_name="inputs:intensity",
        value=intensity,
    )
    color = (
        random.uniform(0.25, 1.0),
        random.uniform(0.25, 1.0),
        random.uniform(0.25, 1.0),
    )
    prim_utils.set_prim_attribute_value(
        prim_path="/World/lightDistant",
        attribute_name="inputs:color",
        value=color,
    )
    angle = random.uniform(1.0, 10.0)
    prim_utils.set_prim_attribute_value(
        prim_path="/World/lightDistant",
        attribute_name="inputs:angle",
        value=angle,
    )

    intensity = random.uniform(0, 2000)
    prim_utils.set_prim_attribute_value(
        prim_path="/World/lightDome",
        attribute_name="inputs:intensity",
        value=intensity,
    )
    color = (
        random.uniform(0.25, 1.0),
        random.uniform(0.25, 1.0),
        random.uniform(0.25, 1.0),
    )
    prim_utils.set_prim_attribute_value(
        prim_path="/World/lightDome",
        attribute_name="inputs:color",
        value=color,
    )


def random_material(mats):
    """随机绑定材质"""
    mat_ground = random.choice(mats)
    mat_plane = random.choice(mats)

    sim_utils.bind_visual_material("/World/defaultGroundPlane", mat_ground)
    sim_utils.bind_visual_material("/World/Objects/Plane", mat_plane)


import math


def random_camera_pose(cam):
    """随机生成相机位姿"""
    # 设置相机距离范围
    r_range = (1.5, 2.0)
    # 在上半球采样相机位置
    r = random.uniform(*r_range)
    theta = random.uniform(0, 2 * np.pi)
    phi = random.uniform(np.pi / 6, np.pi / 3)
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)

    # 随机生成目标点
    x_t = random.uniform(-0.1, 0.1)
    y_t = random.uniform(-0.1, 0.1)
    z_t = random.uniform(-0.1, 0.1)
    
    # 设置相机位姿
    cam.set_world_poses_from_view(
        eyes=torch.tensor([[x, y, z + 1.0]], device=device),
        targets=torch.tensor([[x_t, y_t, z_t + 1.0]], device=device),
    )


def random_place(objs):
    """随机生成物体位置(任意摆放)"""
    for obj in objs:
        obj: RigidObject
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        quat = math_utils.random_orientation(num=1, device=device)
        pos = torch.tensor(
            [[random.uniform(-0.45, 0.45), random.uniform(-0.45, 0.45), 1.2]],
            device=device,
        )
        obj.write_root_pose_to_sim(torch.concatenate([pos, quat], dim=1))


def random_place2(objs):
    """随机生成物体位置(网格摆放)"""
    grid_size = 4
    grid_spacing = 0.2
    grid_start = -0.3
    available_positions = []

    for i in range(grid_size):
        for j in range(grid_size):
            x = grid_start + i * grid_spacing
            y = grid_start + j * grid_spacing
            available_positions.append((x, y))

    random.shuffle(available_positions)

    for obj, pos in zip(objs, available_positions):
        obj: RigidObject
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        quat = math_utils.random_orientation(num=1, device=device)
        z = 1.2
        obj_pos = torch.tensor([[pos[0], pos[1], z]], device=device)
        obj.write_root_pose_to_sim(torch.concatenate([obj_pos, quat], dim=1))


def random_camera(cameras):
    for cam in cameras:
        cam.reset()
        random_camera_pose(cam)


import os
import glob
import pickle


def main():
    # 读取所有USD文件
    usd_dir = "./usd_bop"
    usd_list = glob.glob(os.path.join(usd_dir, "*/*.usd"))

    # 初始化仿真环境
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # 设计场景并添加asset
    cameras, objects, mats = design_scene(other_thing=usd_list)

    # 启动仿真
    sim.reset()
    
    for i, camera in enumerate(cameras):
        camera: Camera
        camera.reset()
        camera.set_intrinsic_matrices(intrinsic_matrix.unsqueeze(0).clone())
        random_camera(cameras)

    print("[INFO]: Setup complete...")

    count = 0
    # 模拟仿真
    while simulation_app.is_running():
        if count % 50 == 0:
            if count > 0:
                saveImg(cameras, objects)
            random_light()
            random_camera(cameras)
            random_material(mats)
            random_place(objects)
        
        if data["count"] > 5000:
            break

        sim.step()
        update_obj(objects)

        count += 1
    
    rgb_data = np.array(data["rgb"])
    id_data = np.array(data["id"])
    np.savez("data.npz", rgb=rgb_data, id=id_data)
    print("[INFO]: Data saved as data.npz")
    pickle.dump(data["info"], open("info.pkl", "wb"))


if __name__ == "__main__":
    main()
    simulation_app.close()
