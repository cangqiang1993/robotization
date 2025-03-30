import numpy as np
from sklearn.cluster import KMeans


def calculate_anchors(annotation_path, n_anchors=9):
    # 读取游戏标注数据
    wh = []  # 存储所有目标的宽高

    # 示例计算 (需替换为真实数据加载)
    example_wh = np.array([[30, 50], [60, 80], [120, 160]])
    kmeans = KMeans(n_clusters=n_anchors)
    kmeans.fit(example_wh)

    anchors = kmeans.cluster_centers_.astype(int)
    return anchors.sort(axis=0)