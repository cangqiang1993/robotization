# 处理图片集
from pathlib import Path

import cv2

"""寻路图像集"""


class ImagesPathfinding:
    def __init__(self, input_path):
        self.imag_path = input_path

    # 加载单个图像或者图像列表
    def load_images(self):
        """智能加载图像，支持单文件或目录"""
        path = Path(self.imag_path)

        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")

        if path.is_file():
            # 单个文件
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            return img

        elif path.is_dir():
            # 目录中的所有图像
            img_paths = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            if not img_paths:
                raise ValueError(f"目录中没有图像文件: {path}")

            imgs = []
            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                if img is not None:
                    imgs.append(img)

            return imgs[0] if len(imgs) == 1 else imgs  # 返回单图或列表
