# 半自动标注工具
import cv2
import numpy as np


class AutoLabeler:
    def generate_annotations(self, frame, depth):
        """自动生成标注建议"""
        # 1. 使用无监督方法检测障碍
        raw_obstacles = self.geometry_analyzer.detect(depth)

        # 2. 聚类连通区域
        contours, _ = cv2.findContours(
            raw_obstacles.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 3. 生成标注建议框
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]