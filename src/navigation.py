import math
from collections import deque

import pyautogui
import time
import random
import cv2


class Navigator:
    def __init__(self, config, detector):
        self.config = config
        self.detector = detector
        self.move_keys = config['3d_navigation']['move_keys']  # 移动按键
        self.move_duration = config['3d_navigation']['move_duration']  # 控制单次移动操作的持续时间（秒），影响移动距离和流畅度
        self.color_ranges = {
            'waypoint': config['minimap']['color_ranges']['waypoint'],
            'obstacle': config['minimap']['color_ranges']['obstacle'],
            'player': config['minimap']['color_ranges']['player']
        }
        self.path_history = deque(maxlen=50)  # 存储位置历史
        self.last_positions = deque(maxlen=5)  # 短期位置记录
        self.stuck_timer = 0

    def execute(self, frame, detections):
        """执行导航逻辑"""
        # 检查是否有路径标记
        markers = [d for d in detections
                   if d['class_name'] in self.config['navigation']['marker_classes']]

        if markers:
            self.follow_markers(markers, frame.shape)
        else:
            self.explore_randomly()

        current_pos = self._get_character_position(frame)
        self.last_positions.append(current_pos)

        if self._is_stuck():
            self._execute_escape_plan()

    def _get_character_position(self, frame):
        """通过UI元素定位角色位置"""
        # 实现根据小地图或角色标记定位
        return (frame.shape[1] // 2, frame.shape[0] // 2)  # 默认为屏幕中心

    def _is_stuck(self):
        """检测是否卡住"""
        if len(self.last_positions) < 5:
            return False

        # 计算移动距离
        total_move = sum(
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(self.last_positions, list(self.last_positions)[1:])
        )
        return total_move < 10  # 5帧内移动小于10像素视为卡住

    def _execute_escape_plan(self):
        """执行脱困策略"""
        # 1. 尝试后退
        pyautogui.keyDown('s')
        time.sleep(1)
        pyautogui.keyUp('s')

        # 2. 随机转向
        pyautogui.moveRel(random.randint(-200, 200), 0, duration=0.5)

        # 3. 跳跃尝试
        pyautogui.press('space')

        self.stuck_counter = 0

    def follow_markers(self, markers, frame_shape):
        """跟随路径标记"""
        # 选择最近的标记
        center_x, center_y = frame_shape[1] // 2, frame_shape[0] // 2

        def distance_to_center(marker):
            x1, y1, x2, y2 = marker['bbox']
            marker_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            return ((marker_center[0] - center_x) ** 2 +
                    (marker_center[1] - center_y) ** 2) ** 0.5

        nearest_marker = min(markers, key=distance_to_center)
        self.move_toward_marker(nearest_marker, frame_shape)

    def move_toward_marker(self, marker, frame_shape):
        """向标记移动"""
        x1, y1, x2, y2 = marker['bbox']
        marker_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        screen_center = (frame_shape[1] // 2, frame_shape[0] // 2)

        # 计算移动方向
        dx = marker_center[0] - screen_center[0]
        dy = marker_center[1] - screen_center[1]

        # 决定移动按键
        if abs(dx) > abs(dy):
            key = 'd' if dx > 0 else 'a'
        else:
            key = 's' if dy > 0 else 'w'

        # 执行移动
        duration = self.move_duration * (1 + random.uniform(-0.2, 0.2))
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def explore_randomly(self):
        """随机探索"""
        key = random.choice(self.move_keys)
        duration = self.move_duration * (1 + random.uniform(0.5, 1.5))

        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

        # 随机转向
        if random.random() < 0.3:
            time.sleep(0.2)
            pyautogui.moveRel(
                random.randint(-100, 100),
                random.randint(-50, 50),
                duration=0.3
            )