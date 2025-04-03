import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VideoFrameLoader")

"""加载和管理预录制的视频帧序列，支持按需读取、缓存和动态调整"""
class VideoFrameLoader:
    def __init__(
            self,
            frame_dir: str,
            cache_size: int = 50,
            target_size: Optional[tuple] = (1280, 720)
    ):
        """
        初始化视频帧加载器

        参数:
            frame_dir: 视频帧存放目录路径
            cache_size: 内存中缓存的帧数量 (默认10帧)
            target_size: 统一调整的帧尺寸 (None表示保持原尺寸)
        """
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        self.frame_paths = []
        for ext in extensions:
            self.frame_paths.extend(Path(frame_dir).glob(ext))
        self.frame_paths = sorted(self.frame_paths)
        if not self.frame_paths:
            raise FileNotFoundError(f"目录中未找到JPG帧: {frame_dir}")

        self.cache_size = cache_size
        self.target_size = target_size
        self._frame_cache = {}  # 帧缓存字典 {index: frame}
        self._current_idx = 0

        logger.info(f"初始化完成，共加载 {len(self.frame_paths)} 帧 (缓存大小: {cache_size})")

    def get_frame(self, index: Optional[int] = None) -> Optional[np.ndarray]:
        """
        获取指定索引的帧 (支持负数索引)

        参数:
            index: 帧索引 (None表示按顺序获取下一帧)
        返回:
            OpenCV图像 (BGR格式) 或 None(读取失败时)
        """
        if index is None:
            index = self._current_idx
            self._current_idx += 1

        # 处理负数索引
        if index < 0:
            index = len(self.frame_paths) + index

        # 边界检查
        if not (0 <= index < len(self.frame_paths)):
            logger.warning(f"索引越界: {index} (总帧数: {len(self.frame_paths)})")
            return None

        # 尝试从缓存获取
        if index in self._frame_cache:
            return self._frame_cache[index]

        # 从磁盘加载
        frame_path = self.frame_paths[index]
        try:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"无法读取图像: {frame_path}")

            # 调整尺寸
            if self.target_size:
                frame = cv2.resize(frame, self.target_size)

            # 更新缓存
            self._update_cache(index, frame)
            return frame

        except Exception as e:
            logger.error(f"加载帧失败 [{index}]: {str(e)}")
            return None

    def _update_cache(self, index: int, frame: np.ndarray):
        """更新帧缓存 (LRU策略)"""
        if len(self._frame_cache) >= self.cache_size:
            # 移除最久未使用的帧
            oldest_idx = min(self._frame_cache.keys())
            del self._frame_cache[oldest_idx]
        self._frame_cache[index] = frame

    def get_all_frames(self) -> Generator[np.ndarray, None, None]:
        """生成器: 按顺序迭代所有帧"""
        for i in range(len(self.frame_paths)):
            frame = self.get_frame(i)
            if frame is not None:
                yield frame

    def get_batch(self, start: int, end: int) -> List[np.ndarray]:
        """获取帧区间 [start, end) 的批量帧"""
        return [self.get_frame(i) for i in range(start, end)
                if self.get_frame(i) is not None]

    @property
    def total_frames(self) -> int:
        """返回总帧数"""
        return len(self.frame_paths)

    @property
    def current_position(self) -> int:
        """返回当前帧索引位置"""
        return self._current_idx

    def reset(self):
        """重置读取位置到开头"""
        self._current_idx = 0
        logger.debug("读取位置已重置")