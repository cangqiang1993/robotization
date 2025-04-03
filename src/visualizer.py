#  调试可视化工具
import cv2
import numpy as np


class NavigationVisualizer:
    def show_debug(self, frame, depth, obstacles):
        """显示调试信息"""
        # 合并显示
        vis = np.hstack([
            frame,
            cv2.cvtColor(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET),
            cv2.cvtColor(obstacles*255, cv2.COLOR_GRAY2BGR)
        ])
        cv2.imshow('Navigation Debug', vis)

    def show_terrain_profile(profile):
        """显示地形特征"""
        # 生成颜色样本
        hsv_range = profile['visual']['color_range']
        sample = np.zeros((100, 100, 3), dtype=np.uint8)
        sample[:, :] = [(hsv_range['h'][0] + hsv_range['h'][1]) // 2,
                        (hsv_range['s'][0] + hsv_range['s'][1]) // 2,
                        (hsv_range['v'][0] + hsv_range['v'][1]) // 2]
        sample = cv2.cvtColor(sample, cv2.COLOR_HSV2BGR)

        # 显示参数
        cv2.putText(sample, f"Friction: {profile['base_params']['friction']}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imshow("Terrain Preview", sample)