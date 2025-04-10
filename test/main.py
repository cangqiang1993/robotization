import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml
from pandas.core import frame

from src.bot import GameBot
from src.detector import YOLODetector, load_config

# 全局变量定义
CONFIG_PATH = "F:/PythonProject/ZhuXIanShiJie/game_auto/config/settings.yaml"  # 默认配置文件路径
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler('output.log', encoding='utf-8')  # 文件输出
    ]
)

def load_images(input_path):
    """智能加载图像，支持单文件或目录"""
    path = Path(input_path)

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


if __name__ == '__main__':
    # try:
        print(torch.__version__)  # 如 '2.0.1+cu117'
        print(torchvision.__version__)  # 如 '0.15.2+cu117'
        print(torch.cuda.is_available())  # 必须返回 True
        # 示例调用 - 替换为您的实际路径
        # result = load_images("F:/PythonProject/ZhuXIanShiJie/game_auto/data/dataset/images/train/walk_circuit")
        #
        # # 处理返回结果（单图或列表）
        # if isinstance(result, list):
        #     print(f"加载了 {len(result)} 张图像")
        #     display_img = result[0]  # 默认显示第一张
        # else:
        #     print("加载了单张图像")
        #     display_img = result
        #
        # # 显示图像（确保是numpy数组）
        # if isinstance(display_img, np.ndarray):
        #     cv2.imshow("Loaded Image", display_img)
        #     cv2.waitKey(3000)
        #     cv2.destroyAllWindows()
        # else:
        #     raise ValueError("图像格式错误，无法显示")
        #
        # config = load_config("F:/PythonProject/ZhuXIanShiJie/game_auto/config/settings.yaml")
        # # 初始化检测器
        # detector = YOLODetector(config)
        #
        # # 统一处理输入（单图或列表都可用）
        # detections = detector.detect(result if isinstance(result, list) else [result])
        # logging.info("检测结果:", detections)

        bot = GameBot("F:/PythonProject/ZhuXIanShiJie/game_auto/config/settings.yaml", "F:/PythonProject/ZhuXIanShiJie/game_auto/data/dataset/images/train")
        bot.start()
        # bot.load_video_frames(result if isinstance(result, list) else [result])
        # global_path = bot.plan_route()
    #
    # except Exception as e:
    #     print(f"错误: {e}")