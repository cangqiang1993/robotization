from typing import List, Dict

import cv2
import numpy as np

from src.mmo.perception_3d import GeometryAnalyzer
from src.navigation import Navigator


class MinimapNavigator:
    def parse_minimap(self, minimap_img):
        """解析小地图信息"""
        # 识别地图标记、路径点、队友位置等
        pass

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
        from src.hybrid_navigator import HybridController  # 动态导入
        self.minimap_processor = MinimapProcessor(config)
        self.terrain_memory = TerrainMemory()
        self.geometryAnalyzer = GeometryAnalyzer(config)

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

    def pathfinding_3d(self, target_pos, safety_distance=None):
        """动态避障"""
        if len(self.obstacles) > 0:
            closest = min(self.obstacles, key=lambda x: x['distance'])
            if closest['distance'] < safety_distance:
                # 计算规避方向
                escape_vector = self._calculate_escape_vector(closest)
                self.move_in_direction(escape_vector)

    def _detect_obstacles(self, frame):
        # 获取融合后的障碍物掩码
        obstacle_mask = self.geometryAnalyzer.detect_obstacles(frame)

        # 转换为检测结果格式
        contours, _ = cv2.findContours(
            obstacle_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        obstacles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            obstacles.append({
                'bbox': [x, y, x + w, y + h],
                'class_name': 'obstacle',
                'confidence': 1.0,
                'position_3d': self.perception_3d.get_3d_position([x, y, x + w, y + h])
            })

        return obstacles

    def explore_3d(self):
        """3D环境探索"""
        # 基于地形记忆的智能探索
        if self.terrain_memory.unexplored_areas:
            direction = self.terrain_memory.get_best_exploration_direction()
            self.move_in_direction(direction)
        else:
            super().explore_randomly()

    def execute_3d_navigation(self, path):
        for step in path:
            while not self._reached_position(step):
                frame = self.obs.get_frame()
                obstacles = self._detect_obstacles(frame)

                # 反应式避障（hybrid_navigator.py的逻辑）
                if obstacles:
                    escape_vector = self._calculate_escape_vector(obstacles)
                    self.move_in_direction(escape_vector)
                else:
                    self.move_to(step)

    def move_to(self, target):
        while not self.reached(target):
            frame = self.obs.get_frame()

            # 步骤1：获取混合决策
            actions = self._avoid_obstacles(frame)

            # 步骤2：执行动作（示例）
            if 'turn_left' in actions:
                self._send_key('a', duration=0.2)
            elif 'move_right' in actions:
                self._send_key('d', duration=0.5)

            # 步骤3：更新权重（每10帧）
            if self.frame_count % 10 == 0:
                self.hybrid_ctrl.update_weights(self.detector.last_detections)


# 反应式避障控制
class ReactiveController:
    def avoid_obstacles(self, obstacle_mask):
        """实时避障决策"""
        # 划分危险区域（左/中/右）
        h, w = obstacle_mask.shape
        sectors = {
            'left': obstacle_mask[h // 2:, :w // 3],
            'center': obstacle_mask[h // 2:, w // 3:2 * w // 3],
            'right': obstacle_mask[h // 2:, 2 * w // 3:]
        }

        # 计算各区域障碍密度
        danger_level = {k: np.mean(v) for k, v in sectors.items()}

        # 生成避障指令
        if danger_level['center'] > 0.3:
            if danger_level['left'] < danger_level['right']:
                return ('turn', -30)  # 向左转
            else:
                return ('turn', 30)  # 向右转
        elif danger_level['left'] > 0.4:
            return ('move', 'right')
        elif danger_level['right'] > 0.4:
            return ('move', 'left')
        return None