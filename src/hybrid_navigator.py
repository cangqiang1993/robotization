# 动态权重调整
import numpy as np


class HybridController:
    def update_weights(self):
        """根据路径检测置信度调整权重"""
        if self.navigator.last_path_confidence > 0.7:
            self.weights = {'visual': 0.8, 'minimap': 0.2}
        else:
            self.weights = {'visual': 0.5, 'minimap': 0.5}

    #冲突解决机制
    def resolve_conflicts(self, yolo_dets, geo_dets):
        """解决监督与非监督结果的冲突"""
        resolved = []
        for yolo in yolo_dets:
            # 优先采用高置信度标注结果
            if yolo['confidence'] > 0.7:
                resolved.append(yolo)
            else:
                # 与几何检测结果交叉验证
                if self._is_geo_confirmed(yolo, geo_dets):
                    resolved.append(yolo)
        return resolved


