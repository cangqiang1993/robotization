import cv2
import numpy as np
import time
from typing import Optional, Tuple, Union
import logging
import yaml
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OBS Capture")


def load_config(config_path: str = None) -> dict:
    """加载YAML配置文件"""
    if config_path is None:
        config_path = Path("settings.yaml")
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


class OBSCapture:
    def __init__(self, config):
        """初始化捕获器"""
        self.config = config
        self.mode = config['obs']['mode']
        self.camera_index = config['obs']['camera_index']
        self.target_fps = config['performance']['target_fps']
        self.last_frame_time = 0
        self.cap = None
        self._init_capture()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3

    def _init_capture(self):
        """初始化虚拟摄像头捕获"""
        try:
            if self.mode == 'virtual_cam':
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    raise RuntimeError(f"无法打开虚拟摄像头，索引: {self.camera_index}")

                # 设置分辨率（可选）
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                logger.info("OBS虚拟摄像头初始化成功")
        except Exception as e:
            logger.error(f"捕获初始化失败: {str(e)}")
            self.release()
            raise

    def get_frame(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """从虚拟摄像头获取帧"""
        try:
            # 控制帧率
            current_time = time.time()
            if (current_time - self.last_frame_time) < (1.0 / self.target_fps):
                return None
            self.last_frame_time = current_time

            # 从摄像头读取帧
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("从虚拟摄像头读取帧失败")
                return None

            # 裁剪指定区域
            if region is not None:
                x, y, w, h = region
                frame = frame[y:y + h, x:x + w]

            return frame

        except Exception as e:
            logger.error(f"获取帧时出错: {str(e)}")
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                self._reconnect()
                return self.get_frame(region)
            return None

    def _reconnect(self):
        """重新连接摄像头"""
        logger.info(f"尝试重连({self.reconnect_attempts}/{self.max_reconnect_attempts})")
        self.release()
        time.sleep(1)
        self._init_capture()

    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("视频捕获资源已释放")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


if __name__ == "__main__":
    try:
        config = load_config("F:/PythonProject/ZhuXIanShiJie/game_auto/config/settings.yaml")
        with OBSCapture(config) as cap:
            while True:
                frame = cap.get_frame()
                if frame is not None:
                    cv2.imshow('OBS Virtual Camera', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
    except Exception as e:
        logger.error(f"程序异常: {str(e)}")
    finally:
        cv2.destroyAllWindows()