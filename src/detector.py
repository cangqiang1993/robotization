from collections import deque
from datetime import time

import cv2
import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将坐标从img1_shape缩放到img0_shape
    参数:
        img1_shape: 模型输入尺寸 (height, width)
        coords: 原始坐标 [x1,y1,x2,y2]
        img0_shape: 原始图像尺寸 (height, width)
        ratio_pad: 可选的缩放比例和填充值
    返回:
        缩放后的坐标
    """
    if ratio_pad is None:  # 计算缩放比例和填充
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 增益
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 填充
    else:
        gain, pad = ratio_pad

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain  # 缩放回原图尺寸

    # 裁剪坐标到图像边界内
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1,x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1,y2
    return coords

class YOLODetector:
    def __init__(self, weights_path='data/weights/best.pt', device=''):
        # 设备选择
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = torch.jit.load(weights_path) if weights_path.endswith('.torchscript') else \
            attempt_load(weights_path, map_location=self.device)
        self.model.eval()

        # 类别名称
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        self.cache_size = 3  # 缓存最近3帧
        self.frame_cache = deque(maxlen=self.cache_size)
        self.detection_cache = deque(maxlen=self.cache_size)
        self.adaptive_sizes = [640, 480, 320]  # 动态调整的检测尺寸
        self.current_size_idx = 0

    def detect_3d(self, image, frame_info=None):
        """支持3D位置信息的检测"""
        detections = self.detect(image)

        if frame_info and hasattr(frame_info, 'depth_map'):
            for det in detections:
                # 添加3D位置信息
                det['position_3d'] = self._calculate_3d_position(
                    det['bbox'],
                    frame_info.depth_map
                )
        return detections

    def _calculate_3d_position(self, bbox, depth_map):
        """基于深度图计算3D位置"""
        x_center = (bbox[0] + bbox[2]) // 2
        y_center = (bbox[1] + bbox[3]) // 2
        depth = depth_map[y_center, x_center]

        # 转换为游戏内坐标 (需根据游戏校准)
        return {
            'x': (x_center - self.camera_center[0]) * depth / self.focal_length,
            'y': (y_center - self.camera_center[1]) * depth / self.focal_length,
            'z': depth,
            'distance': depth
        }

    def detect(self, image, conf_thres=0.7, iou_thres=0.45):
        # 缓存检查
        for cached_frame, cached_dets in zip(self.frame_cache, self.detection_cache):
            if self._frame_similarity(image, cached_frame) > 0.9:
                return cached_dets

        # 动态调整检测尺寸
        img_size = self._get_adaptive_size()
        img = self.preprocess(image, img_size)

        """
        执行目标检测
        返回: [x1, y1, x2, y2, confidence, class_id]
        """
        # 图像预处理
        img = self.preprocess(image, img_size)

        # 推理
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]

        # NMS处理
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # 后处理
        results = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape[:2]).round()
                for *xyxy, conf, cls in det:
                    results.append({
                        'bbox': [int(x) for x in xyxy],
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': self.names[int(cls)]
                    })

        # 更新缓存
        self.frame_cache.append(image.copy())
        self.detection_cache.append(results)
        return results

    def _frame_similarity(self, frame1, frame2):
        """计算两帧相似度(0-1)"""
        if frame1.shape != frame2.shape:
            return 0
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        return cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]

    def _get_adaptive_size(self):
        """根据帧率动态调整检测尺寸"""
        if len(self.detection_cache) < 2:
            return self.adaptive_sizes[0]

        process_time = time.time() - self.last_detect_time
        if process_time > 0.1:  # 处理时间过长则降级
            self.current_size_idx = min(self.current_size_idx + 1, len(self.adaptive_sizes) - 1)
        elif process_time < 0.05:  # 处理很快则升级
            self.current_size_idx = max(self.current_size_idx - 1, 0)
        return self.adaptive_sizes[self.current_size_idx]
    def preprocess(self, img, img_size=640):
        """图像预处理"""
        # 调整大小和填充
        img = letterbox(img, img_size, stride=self.model.stride)[0]

        # 转换为tensor
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # 归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img