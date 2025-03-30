from typing import List, Dict

import cv2
import numpy as np

from src.navigation import Navigator


class MinimapProcessor:
    """小地图处理器，用于解析游戏小地图信息"""

    def __init__(self, config: dict):
        self.config = config['minimap']
        self.waypoints = []

        # 颜色范围配置 (HSV格式)
        self.color_ranges = {
            'waypoint': ([30, 150, 100], [50, 255, 255]),  # 黄色路径点
            'obstacle': ([0, 100, 100], [10, 255, 255]),  # 红色障碍物
            'player': ([100, 150, 150], [140, 255, 255])  # 蓝色玩家标记
        }

    def update(self, minimap_image: np.ndarray) -> None:
        """更新小地图信息"""
        # 预处理图像
        hsv = cv2.cvtColor(minimap_image, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 检测路径点
        self.waypoints = self._detect_waypoints(blurred)

    def _detect_waypoints(self, hsv_image: np.ndarray) -> List[Dict]:
        """检测小地图上的路径点"""
        lower, upper = self.color_ranges['waypoint']
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        waypoints = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:  # 过滤小噪点
                x, y, w, h = cv2.boundingRect(cnt)
                waypoints.append({
                    'position': (x + w // 2, y + h // 2),  # 中心坐标
                    'size': max(w, h)  # 标记大小
                })

        return waypoints

    def get_waypoints(self) -> List[Dict]:
        """获取当前路径点列表"""
        return self.waypoints

    def clear_waypoints(self) -> None:
        """清空路径点缓存"""
        self.waypoints = []


class TerrainMemory:
    """地形记忆系统，记录探索过的区域和障碍物"""

    def __init__(self):
        self.explored_areas = []
        self.obstacles = []
        self.unexplored_areas = []

    def update(self, obstacles: List) -> None:
        """更新地形记忆"""
        self.obstacles = obstacles
        self._update_exploration_map()

    def _update_exploration_map(self) -> None:
        """更新探索区域地图（简化版）"""
        # 这里应该实现更复杂的地图记忆算法
        # 示例仅记录最近的障碍物位置
        new_unexplored = []
        for obs in self.obstacles:
            if obs not in self.explored_areas:
                new_unexplored.append(obs)

        self.unexplored_areas = new_unexplored

    def get_best_exploration_direction(self) -> tuple:
        """获取最佳探索方向"""
        if not self.unexplored_areas:
            return (0, 0)  # 默认方向

        # 简单实现：选择最近的未探索区域
        closest = min(self.unexplored_areas,
                      key=lambda x: x.get('distance', float('inf')))
        return closest.get('direction', (1, 0))  # 默认向右移动


class MMO_Navigator(Navigator):
    def __init__(self, config, detector):
        super().__init__(config, detector)
        self.minimap_processor = MinimapProcessor(config)
        self.terrain_memory = TerrainMemory()

    def execute(self, frame, detections):
        # 处理小地图
        self.update_minimap(frame)

        # 3D环境分析
        obstacles = self._detect_obstacles(frame)
        self.terrain_memory.update(obstacles)

        # 智能路径规划
        if self.waypoints:
            success = self.follow_waypoints()
            if not success:
                self._find_alternative_path()
        else:
            self.explore_3d()

    def _detect_obstacles(self, frame):
        """检测3D障碍物"""
        detections = self.detector.detect_3d(frame)
        return [d for d in detections if self._is_obstacle(d)]

    def explore_3d(self):
        """3D环境探索"""
        # 基于地形记忆的智能探索
        if self.terrain_memory.unexplored_areas:
            direction = self.terrain_memory.get_best_exploration_direction()
            self.move_in_direction(direction)
        else:
            super().explore_randomly()