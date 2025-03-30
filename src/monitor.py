# monitor.py
import json
import time

import cv2
import numpy as np


class GameMonitor:
    def __init__(self, config):
        self.config = config
        self.performance_log = []
        self.status = {
            'fps': 0,
            'detection_time': 0,
            'combat_state': 'idle'
        }

    def update_status(self, **kwargs):
        self.status.update(kwargs)
        self._log_performance()

    def draw_overlay(self, frame):
        """在帧上绘制状态面板"""
        h, w = frame.shape[:2]
        overlay = np.zeros((100, w, 3), dtype=np.uint8)

        # 基础信息
        texts = [
            f"FPS: {self.status['fps']:.1f}",
            f"Mode: {self.status.get('mode', 'unknown')}",
            f"Det: {self.status['detection_time'] * 1000:.1f}ms"
        ]

        for i, text in enumerate(texts):
            cv2.putText(overlay, text, (10, 30 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # CPU/GPU使用率条
        if 'cpu_usage' in self.status:
            cv2.rectangle(overlay, (200, 10), (200 + int(self.status['cpu_usage'] * 2), 30),
                          (0, 0, 255), -1)

        return np.vstack([overlay, frame])

    def _log_performance(self):
        """记录性能数据"""
        self.performance_log.append({
            'timestamp': time.time(),
            'data': self.status.copy()
        })
        if len(self.performance_log) > 100:
            self._save_log()

    def _save_log(self):
        """保存日志到文件"""
        filename = f"logs/performance_{time.strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(self.performance_log, f)
        self.performance_log.clear()