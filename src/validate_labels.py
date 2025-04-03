# 标注验证脚本
from pathlib import Path
import cv2
import numpy as np


def count_labeled_area(label_file, img_file=None):
    """计算标注文件中已标注的障碍物区域面积（基于YOLO格式）
    Args:
        label_file (Path): 标注文件路径（.txt）
        img_file (Path): 对应的图片文件路径（可选，用于动态获取尺寸）
    Returns:
        int: 已标注的像素总数（矩形区域面积之和）
    """
    if not label_file.exists():
        return 0  # 如果标注文件不存在，返回0

    # 动态获取图像尺寸
    if img_file is None:
        img_file = label_file.with_suffix('.jpg')  # 默认同级目录同名的.jpg文件
    try:
        img = cv2.imread(str(img_file))
        if img is None:
            raise FileNotFoundError(f"图片文件不存在: {img_file}")
        img_height, img_width = img.shape[:2]
    except Exception as e:
        print(f"警告: 无法读取图片尺寸，使用默认值 1000x1000。错误: {e}")
        img_width, img_height = 1000, 1000  # 备用默认值

    total_area = 0
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # 跳过无效行

            # 解析YOLO格式：class_id x_center y_center width height
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            # 计算矩形面积并累加
            total_area += int(width * height)

    return total_area


def check_annotations(dataset_path, depth_estimator, geometry_analyzer):
    """验证标注质量
    Args:
        dataset_path (str): 数据集路径
        depth_estimator: 深度估计器对象，需实现 `estimate(img)` 方法
        geometry_analyzer: 几何分析器对象，需实现 `detect(depth)` 方法
    """
    for img_file in Path(dataset_path).glob('*.jpg'):
        label_file = img_file.with_suffix('.txt')

        # 检查标注是否覆盖障碍物
        img = cv2.imread(str(img_file))
        depth = depth_estimator.estimate(img)
        obstacles = geometry_analyzer.detect(depth)

        # 计算未标注的障碍物比例（传入img_file以动态获取尺寸）
        labeled_area = count_labeled_area(label_file, img_file)
        unlabeled = np.sum(obstacles) - labeled_area
        if unlabeled > obstacles.size * 0.1:
            print(f"警告: {img_file} 存在未标注障碍")