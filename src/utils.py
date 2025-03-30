import cv2
import numpy as np
import time
import logging
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path
import yaml
import json
from dataclasses import dataclass
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GameUtils")


@dataclass
class FrameInfo:
    """图像帧信息数据类"""
    frame: np.ndarray
    timestamp: float
    source: str


class ColorRange(Enum):
    """颜色范围枚举"""
    RED = ((0, 100, 100), (10, 255, 255))  # HSV范围
    BLUE = ((100, 100, 100), (140, 255, 255))
    GREEN = ((40, 100, 100), (80, 255, 255))


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    加载YAML配置文件
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise


def save_config(config: Dict, config_path: Union[str, Path]):
    """
    保存配置到YAML文件
    :param config: 配置字典
    :param config_path: 保存路径
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
    except Exception as e:
        logger.error(f"保存配置文件失败: {str(e)}")
        raise


def resize_with_aspect_ratio(
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        inter: int = cv2.INTER_AREA
) -> np.ndarray:
    """
    保持宽高比调整图像大小
    :param image: 输入图像
    :param width: 目标宽度 (None表示自动计算)
    :param height: 目标高度 (None表示自动计算)
    :param inter: 插值方法
    :return: 调整后的图像
    """
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def extract_roi(
        image: np.ndarray,
        roi: Tuple[int, int, int, int],
        padding: int = 0
) -> np.ndarray:
    """
    提取图像中的感兴趣区域(ROI)
    :param image: 输入图像
    :param roi: (x, y, w, h)格式的区域
    :param padding: 扩展像素数
    :return: ROI图像
    """
    x, y, w, h = roi
    h_img, w_img = image.shape[:2]

    # 计算带padding的ROI
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)

    return image[y1:y2, x1:x2]


def find_color_mask(
        image: np.ndarray,
        color_range: ColorRange,
        blur_kernel: Tuple[int, int] = (5, 5),
        threshold: int = 100
) -> np.ndarray:
    """
    根据颜色范围创建掩码
    :param image: 输入图像(BGR格式)
    :param color_range: 颜色范围枚举
    :param blur_kernel: 高斯模糊核大小
    :param threshold: 二值化阈值
    :return: 二值化掩码
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, blur_kernel, 0)

    lower, upper = color_range.value
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return mask


def draw_detections(
        image: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制检测结果
    :param image: 原始图像
    :param detections: 检测结果列表 [{'bbox': [x1,y1,x2,y2], ...}]
    :param color: 绘制颜色 (BGR)
    :param thickness: 线宽
    :return: 绘制后的图像
    """
    result = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.get('class_name', '')} {det.get('confidence', 0):.2f}"
        cv2.putText(result, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    计算帧率
    :param start_time: 开始时间(time.time())
    :param frame_count: 帧计数
    :return: 当前FPS
    """
    elapsed = time.time() - start_time
    return frame_count / elapsed if elapsed > 0 else 0


def timeit(func):
    """函数执行时间测量装饰器"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"{func.__name__} 执行时间: {end - start:.4f}s")
        return result

    return wrapper


class FPSCounter:
    """FPS计算器类"""

    def __init__(self, window_size: int = 30):
        self.times = []
        self.window_size = window_size

    def update(self):
        """更新计数器"""
        self.times.append(time.time())
        if len(self.times) > self.window_size:
            self.times.pop(0)

    def get_fps(self) -> float:
        """获取当前FPS"""
        if len(self.times) < 2:
            return 0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


def save_image(
        image: np.ndarray,
        path: Union[str, Path],
        quality: int = 95
) -> bool:
    """
    保存图像文件
    :param image: 要保存的图像
    :param path: 保存路径
    :param quality: JPEG质量(1-100)
    :return: 是否成功
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        else:
            cv2.imwrite(str(path), image)
        return True
    except Exception as e:
        logger.error(f"保存图像失败: {str(e)}")
        return False


def read_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    读取图像文件
    :param path: 图像路径
    :return: OpenCV图像或None
    """
    try:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("无法读取图像文件")
        return image
    except Exception as e:
        logger.error(f"读取图像失败: {str(e)}")
        return None


def compare_frames(
        frame1: np.ndarray,
        frame2: np.ndarray,
        method: str = 'mse'
) -> float:
    """
    比较两帧图像的相似度
    :param frame1: 第一帧
    :param frame2: 第二帧
    :param method: 比较方法 ('mse', 'ssim', 'hist')  SSIM 计算较耗时，适合对质量要求高的场景、实时处理建议使用 Hist 或 MSE
    :return: 相似度分数 (数值越大表示越相似，除mse外)
    """
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    if method == 'mse':
        # 均方误差 (值越小越相似)
        err = np.sum((frame1.astype("float") - frame2.astype("float")) ** 2)
        err /= float(frame1.shape[0] * frame1.shape[1])
        return err
    elif method == 'ssim':
        # 结构相似性 (0-1, 1表示完全相同)
        from skimage.metrics import structural_similarity
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        score, _ = structural_similarity(gray1, gray2, full=True)
        return score
    elif method == 'hist':
        # 直方图相关性 (-1到1, 1表示完全匹配)
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    else:
        raise ValueError(f"未知的比较方法: {method}")


def create_video_writer(
        output_path: Union[str, Path],
        frame_size: Tuple[int, int],
        fps: int = 30,
        codec: str = 'XVID'
) -> Optional[cv2.VideoWriter]:
    """
    创建视频写入器
    :param output_path: 输出路径
    :param frame_size: 帧大小 (width, height)
    :param fps: 帧率
    :param codec: 编码器
    :return: VideoWriter对象
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    except Exception as e:
        logger.error(f"创建视频写入器失败: {str(e)}")
        return None


def mouse_callback(event, x, y, flags, param):
    """OpenCV鼠标回调示例"""
    if event == cv2.EVENT_LBUTTONDOWN:
        logger.info(f"鼠标点击位置: ({x}, {y})")
        param['click_pos'] = (x, y)


def interactive_select_roi(image: np.ndarray, window_name: str = "Select ROI") -> Optional[Tuple[int, int, int, int]]:
    """
    交互式选择ROI区域
    :param image: 输入图像
    :param window_name: 窗口名称
    :return: (x, y, w, h) 或 None
    """
    params = {'roi': None, 'drawing': False, 'ix': -1, 'iy': -1}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['drawing'] = True
            param['ix'], param['iy'] = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if param['drawing']:
                img_copy = image.copy()
                cv2.rectangle(img_copy, (param['ix'], param['iy']), (x, y), (0, 255, 0), 2)
                cv2.imshow(window_name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            param['drawing'] = False
            x1, y1 = min(param['ix'], x), min(param['iy'], y)
            x2, y2 = max(param['ix'], x), max(param['iy'], y)
            param['roi'] = (x1, y1, x2 - x1, y2 - y1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(window_name, image)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, params)
    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or params['roi'] is not None:
            break

    cv2.destroyAllWindows()
    return params['roi']


# 使用示例
if __name__ == "__main__":
    # 示例: 颜色掩码检测
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (0, 0, 255), -1)  # 红色矩形

    mask = find_color_mask(img, ColorRange.RED)
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)