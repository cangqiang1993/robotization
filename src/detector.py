
import os
from collections import deque, defaultdict
from datetime import time
from pathlib import Path

import cv2
import torch
import numpy as np
import yaml
from pandas.core import frame
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox


# 配置日志
# 初始化日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler('output.log', encoding='utf-8')  # 文件输出
    ]
)


# 全局变量定义
CONFIG_PATH = "E:/PythonProject/ZhuXIanShiJie/game_auto/config/settings.yaml"  # 默认配置文件路径

# 初始化配置文件
def load_config(config_path: str = None) -> dict:
    """加载YAML配置文件"""
    if config_path is None:
        config_path = Path(CONFIG_PATH)
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path.absolute()}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_fields = {
        "obs": ["mode"],
        "performance": ["target_fps"]
    }
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"配置缺少必要部分: {section}")
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"配置缺少必要字段: {section}.{field}")

    return config


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
    """改进的坐标缩放函数"""
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, device=img1_shape.device, dtype=torch.float32)

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
    # 确保输出是可合理取整的浮点数
    coords = coords.float()  # 强制转换为浮点
    coords = torch.nan_to_num(coords)  # 处理可能的NaN值

    # 限制坐标范围
    coords[:, [0, 2]] = torch.clamp(coords[:, [0, 2]], 0, img0_shape[1])
    coords[:, [1, 3]] = torch.clamp(coords[:, [1, 3]], 0, img0_shape[0])
    coords.round()
    return coords  # 直接返回已取整的值

class YOLODetector:
    def __init__(self, config):
        # self.config = load_config()
        # 设备选择
        self.device = config['yolo'].get('device')
        self._check_pytorch_version()
        logging.info(f'当前配置设备为: {self.device}')
        logging.info(f'当前显卡型号：{torch.cuda.get_device_name(0)}')
        logging.info(f"CUDA版本号 是否合格：{torch.cuda.is_available()}")  # 应该返回True
        logging.info(f"CUDA版本号:{torch.version.cuda}")  # 应该显示CUDA版本号
        # 获取项目根目录（假设detector.py在src/目录下）
        project_root = Path(__file__).parent.parent  # 回退两级到game_auto/
        # 处理权重路径
        weights_path = Path(config['yolo'].get('weights_path'))
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path

        # 验证路径存在
        if not weights_path.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {weights_path}\n"
                f"当前工作目录: {os.getcwd()}\n"
                f"尝试的绝对路径: {weights_path.absolute()}"
            )

        # 加载模型
        if weights_path.suffix == '.torchscript':  # 使用Path对象的suffix属性
            self.model = torch.jit.load(weights_path)
        else:
            self.model = attempt_load(weights_path)
            self.model.to(self.device)

        self.model.eval()
        logging.info(f"加载模型路径: {weights_path}")
        # 类别名称
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.cache_size = 3  # 缓存最近3帧
        self.frame_cache = deque(maxlen=self.cache_size)
        self.detection_cache = deque(maxlen=self.cache_size)
        self.adaptive_sizes = [640, 480, 320]  # 动态调整的检测尺寸
        self.current_size_idx = 0

    def _check_pytorch_version(self):
        """检查PyTorch版本兼容性"""
        version = torch.__version__.split('.')
        major, minor = int(version[0]), int(version[1])
        if major < 1 or (major == 1 and minor < 8):
            logging.warning(f"PyTorch版本 {torch.__version__} 可能不兼容，建议升级到1.8+")

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

    def detect(self, images, conf_thres=0.7, iou_thres=0.45):
        """批量检测图像"""
        if not isinstance(images, list):
            images = [images]

        # 批量预处理
        img_tensors = []
        img_shapes = []
        for img in images:
            img_size = self._get_adaptive_size()
            processed_img = self.preprocess(img, img_size)
            assert isinstance(processed_img, torch.Tensor), f"预处理结果应为张量，实际得到 {type(processed_img)}"
            assert not processed_img.requires_grad, "预处理张量不应需要梯度"
            img_tensors.append(processed_img)
            img_shapes.append(img.shape[:2])

        img_batch = torch.cat(img_tensors, dim=0)

        with torch.no_grad():
            preds = self.model(img_batch, augment=False)[0]

        preds = non_max_suppression(preds, conf_thres, iou_thres)
        batch_results = []
        for i, (det, orig_shape) in enumerate(zip(preds, img_shapes)):
            img_results = []
            if det is not None and len(det):
                # 1. 缩放坐标
                det[:, :4] = scale_coords(img_batch.shape[2:], det[:, :4], orig_shape)
                # 直接转换类型（因为值已预先取整）
                det[:, :4] = det[:, :4].to(torch.int32)
                # 2. 安全转换为整数坐标
                det = self._safe_convert_to_int(det)
                # 3. 转换到CPU
                det = det.cpu().numpy()

                for *xyxy, conf, cls in det:
                    img_results.append({
                        'bbox': [int(x) for x in xyxy],
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': self.names[int(cls)]
                    })

            batch_results.append(img_results)
            self._update_cache(images[i], img_results)
        batch_results = batch_results[0] if len(images) == 1 else batch_results
        return batch_results

    def _safe_convert_to_int(self, det):
        """强制将坐标转换为整数且不保留小数（100% 类型安全版本）"""
        # 1. 提取坐标并确保设备一致
        coords = det[:, :4].clone().to(det.device)

        # 2. 四舍五入并转换为整数（两种等效方式）
        coords = torch.round(coords).to(torch.int32)  # 方法1
        # coords = torch.round(coords).int()         # 方法2

        # 3. 创建新张.量（关键步骤：指定输出类型为整数）
        new_det = torch.empty_like(det, dtype=torch.float32)  # 其他列保持浮点
        new_det[:, :4] = coords.float()  # 临时用浮点存储（后续会覆盖）

        # 4. 精确赋值（保持整数性）
        int_coords = coords.to(torch.int32)
        new_det[:, :4] = int_coords.int()  # 显式类型转换

        # 5. 恢复其他列数据
        new_det[:, 4:] = det[:, 4:]

        # 验证（双重检查）
        assert new_det[:, :4].dtype == torch.float32, "类型错误"
        assert torch.all(new_det[:, :4] == int_coords.float()), "值不匹配"

        return new_det


    def _update_cache(self, image, detections):
        """更新检测缓存"""
        self.frame_cache.append(image.copy())
        self.detection_cache.append(detections)

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
        """完全安全的图像预处理方法"""
        try:
            # 1. 使用letterbox调整大小
            img = letterbox(img, img_size, stride=self.model.stride, auto=False)[0]
            print('获取图片', img)
            # 2. 确保图像是连续的numpy数组
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)

            # 3. BGR转RGB和转置维度 (HWC to CHW)
            img = img[:, :, ::-1]  # BGR to RGB
            img = img.transpose(2, 0, 1)  # HWC to CHW

            # 4. 再次确保数据连续性
            img = np.ascontiguousarray(img)

            # 5. 转换为PyTorch张量
            img = torch.from_numpy(img)

            # 6. 转移到目标设备并归一化
            img = img.to(self.device)
            img = img.float() / 255.0

            # 7. 添加批次维度
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            print(img)
            return img
        except Exception as e:
            logging.error(f"预处理失败: {str(e)}")
            raise RuntimeError(f"图像预处理错误: {str(e)}")


# 障碍物频率统计
class ObstacleLogger:
    def __init__(self):
        self.counter = defaultdict(int)

    def log_detections(self, detections):
        for det in detections:
            if det['class_name'] in ['cliff', 'water', 'rock']:
                self.counter[det['class_name']] += 1

    def get_top_obstacles(self, n=3):
        return sorted(self.counter.items(), key=lambda x: -x[1])[:n]


# 增量学习接口
class OnlineLearner:
    def fine_tune(self, new_samples):
        """增量训练"""
        # 1. 准备新数据
        train_loader = self._create_dataloader(new_samples)

        # 2. 微调模型
        self.model.train()
        for _ in range(5):  # 少量迭代
            self._run_epoch(train_loader)

        # 3. 验证效果
        val_loss = self.validate()
        return val_loss

if __name__ == '__main__':
    from src.detector import YOLODetector

    config = load_config()
    # 初始化检测器
    detector = YOLODetector(config)
    detections = detector.detect(frame)
    print("检测器初始化成功！")