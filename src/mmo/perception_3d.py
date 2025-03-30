import torch
import numpy as np
import cv2
from typing import Optional


class DepthEstimator:
    """深度估计器，用于从2D图像预测深度图"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载预训练深度估计模型"""
        try:
            # 这里应该替换为您实际使用的深度估计模型
            # 示例使用MiDaS小型模型结构
            model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"无法加载深度估计模型: {str(e)}")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        从RGB图像估计深度图
        返回:
            depth_map: 归一化的深度图 (0-1, 1表示最近)
        """
        # 预处理
        img = self._preprocess(frame)

        # 推理
        with torch.no_grad():
            prediction = self.model(img)
            depth = prediction.squeeze().cpu().numpy()

        # 后处理
        depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
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
        self.config = config['3d_perception']
        self.depth_estimator = DepthEstimator(
            model_path=self.config['depth_model'],
            device=self.config.get('device', 'cuda')
        )
        self.camera_params = self.config['camera']

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