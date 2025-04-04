import logging

import torch
import numpy as np
import cv2
from typing import Optional

from yolov5.models import yolo

from src.detector import YOLODetector
from src.images_array import ImagesPathfinding


# 深度图与几何分析的集成

class DepthEstimator:
    """深度估计器，用于从2D图像预测深度图"""

    def __init__(self, config, device: str = 'cuda'):
        self.device = device
        self.model = self._load_model(config['3d_navigation']['depth_model'])
        self.model.eval()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载预训练深度估计模型"""
        try:
            # 这里应该替换为您实际使用的深度估计模型
            # 示例使用MiDaS小型模型结构
            logging.info(f'加载预训练深度估计模型路径： {model_path}')
            model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', pretrained=False)
            model.load_state_dict(torch.load(model_path))
            model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"无法加载深度估计模型: {str(e)}")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        从RGB图像估计深度图
        返回:
            depth_map: 归一化的深度图 (0-1, 1表示最近)，与输入尺寸相同
        """
        original_h, original_w = frame.shape[:2]

        # 预处理
        img = self._preprocess(frame)

        # 推理
        with torch.no_grad():
            prediction = self.model(img)
            depth = prediction.squeeze().cpu().numpy()

        # 后处理
        depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

        # 调整回原始尺寸
        depth = cv2.resize(depth, (original_w, original_h))
        return depth

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """将OpenCV图像转换为模型输入格式"""
        # 调整大小 (示例模型需要384x384输入)
        img = cv2.resize(frame, (384, 384))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 归一化并转换为tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)


class ThreeDPerception:
    def __init__(self, config):
        self.config = config
        self.depth_estimator = DepthEstimator(config)   # 深度模型路径 config['3d_navigation']['depth_model']
        self.geometry_analyzer = GeometryAnalyzer(config)  # 几何分析配置 config['3d_navigation']['geometry']
        self.camera_params = self.config['3d_navigation']['camera']  # 相机参数
        # 地形参数
        self.walkable_threshold = config['3d_navigation']['navigation']['max_slope_angle']  # 最大可行走坡度（度）
        logging.info("最大可行走坡度: {}".format(self.walkable_threshold))
        self.step_height = config['3d_navigation']['navigation']['max_step_height']  # 最大台阶高度（归一化值）
        logging.info("最大台阶高度（归一化值）: {}".format(self.step_height))

    def get_walkable_mask(self, depth_map):
        """生成可行走区域掩码"""
        # 计算法线向量
        normals = self._calculate_normals(depth_map)

        # 通过法线与垂直方向的夹角判断坡度
        vertical = np.array([0, 0, 1])
        angles = np.arccos(np.dot(normals, vertical))

        # 生成可行走区域（坡度<阈值 且 高度差<台阶高度）
        walkable = (angles < np.radians(self.walkable_threshold)) & \
                   (depth_map < self.step_height)

        return walkable.astype(np.uint8)

    def update(self, frame):
        """更新3D环境感知"""
        self.depth_map = self.depth_estimator.estimate(frame)
        self.normals = self._calculate_normals(self.depth_map)

    def get_3d_position(self, bbox):
        """从2D bbox获取3D位置"""
        # 使用深度图中值作为距离估计
        x1, y1, x2, y2 = map(int, bbox)
        crop_depth = self.depth_map[y1:y2, x1:x2]
        median_depth = np.median(crop_depth)

        # 转换为3D坐标
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return {
            'x': (center_x - self.camera_params['cx']) * median_depth / self.camera_params['fx'],
            'y': (center_y - self.camera_params['cy']) * median_depth / self.camera_params['fy'],
            'z': median_depth,
            'screen_pos': (center_x, center_y)
        }

    def _calculate_normals(self, depth_map):
        """计算深度图的表面法线向量

        参数:
            depth_map (np.ndarray): 输入的深度图，值范围0-1（1表示最近）

        返回:
            np.ndarray: 法线向量图，形状为 (H, W, 3)，每个像素是单位法线向量 [nx, ny, nz]
        """
        # 1. 将深度图转换为实际距离（假设深度图已经是线性深度）
        # 注意：这里需要根据实际深度模型调整转换逻辑
        depth = depth_map.copy()

        # 2. 计算梯度（使用Sobel算子）
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)  # 水平梯度
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)  # 垂直梯度

        # 3. 计算法线向量（基于梯度）
        # 法线公式: n = [-dz/dx, -dz/dy, 1]
        normals = np.dstack((-grad_x, -grad_y, np.ones_like(depth)))

        # 4. 归一化为单位向量
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = np.divide(normals, norm, where=(norm != 0))

        return normals


# 深度图与几何分析的集成
class GeometryAnalyzer:
    def __init__(self, config):
        self.max_slope = config['3d_navigation']['geometry']['max_slope']
        self.step_height = config['3d_navigation']['geometry']['step_height']
        self.config = config
        self.detector = YOLODetector(self.config)
        self.ImagesPathfinding = ImagesPathfinding
        self.depth_estimator = DepthEstimator(config)  # 直接初始化深度估计器

    def _calculate_normals(self, depth_map):
        """计算深度图的表面法线向量"""
        # 使用Sobel算子计算梯度
        grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

        # 法线向量 = (-dx, -dy, 1) 并归一化
        normals = np.dstack((-grad_x, -grad_y, np.ones_like(depth_map)))
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = np.divide(normals, norms, where=norms != 0)
        return normals

    def detect_obstacles(self, frame: np.ndarray) -> np.ndarray:
        # 1. YOLO检测语义障碍物
        result = self.ImagesPathfinding(self.config['images_path']['imag_path']).load_images()
        detections = self.detector.detect(result if isinstance(result, list) else [result])
        yolo_obstacles = np.zeros(frame.shape[:2], dtype=np.uint8)

        # 筛选障碍物类别
        obstacle_classes = self.config['game']['navigation'].get('obstacle_classes', ['tree', 'rock', 'wall', 'monster'])

        for det in detections:
            if isinstance(det, dict) and 'class_name' in det:
                class_name = det['class_name']
                bbox = det['bbox']
            elif isinstance(det, (list, tuple)) and len(det) >= 6:
                class_idx = int(det[5])
                class_name = self.detector.names[class_idx]
                bbox = det[:4]
            else:
                continue

            if class_name in obstacle_classes:
                x1, y1, x2, y2 = map(int, bbox)
                yolo_obstacles[y1:y2, x1:x2] = 1

        # 2. 深度模型几何分析（使用自己的 depth_estimator）
        depth_map = self.depth_estimator.estimate(frame)

        # 调用自己的 _calculate_normals 方法
        normals = self._calculate_normals(depth_map)
        vertical = np.array([0, 0, 1])
        slope = np.arccos(np.dot(normals, vertical))
        slope_obstacles = (slope > np.radians(self.max_slope))

        blurred_depth = cv2.GaussianBlur(depth_map, (5, 5), 0)
        height_diff = np.abs(depth_map - blurred_depth)
        step_obstacles = (height_diff > self.step_height)

        geometry_obstacles = slope_obstacles | step_obstacles

        # 融合策略
        yolo_weight = self.config['hybrid_nav'].get('yolo_weight', 0.6)
        combined = yolo_weight * yolo_obstacles + (1 - yolo_weight) * geometry_obstacles.astype(np.float32)
        _, combined_mask = cv2.threshold(combined, 0.5, 1, cv2.THRESH_BINARY)

        # 后处理
        contours, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                cv2.drawContours(combined_mask, [cnt], -1, 0, -1)

        return combined_mask.astype(bool)