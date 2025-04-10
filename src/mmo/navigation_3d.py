from collections import deque
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

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

class EnhancedMinimapProcessor(MinimapProcessor):
    def __init__(self, config: dict):
        super().__init__(config)
        # 路径颜色自适应参数
        self.color_adaptation = {
            'learning_rate': 0.1,
            'color_history': deque(maxlen=100),
            'initial_color_range': self.color_ranges['waypoint']
        }

        # 路径几何特征参数
        self.geometry_params = {
            'min_path_width': 5,  # 最小路径宽度(像素)
            'min_path_length': 20,  # 最小路径长度(像素)
            'aspect_ratio_thresh': 3  # 长宽比阈值
        }

    def update_color_profile(self, confirmed_path_region: np.ndarray):
        """用确认的路径区域更新颜色特征"""
        hsv = cv2.cvtColor(confirmed_path_region, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv, axis=(0, 1))
        self.color_adaptation['color_history'].append(mean_hsv)

        # 动态调整颜色范围
        if len(self.color_adaptation['color_history']) > 10:
            hist_array = np.array(self.color_adaptation['color_history'])
            mean = np.mean(hist_array, axis=0)
            std = np.std(hist_array, axis=0)

            # 更新颜色范围 (HSV格式)
            self.color_ranges['path'] = (
                np.clip(mean - 1.5 * std, 0, 255).astype(int).tolist(),
                np.clip(mean + 1.5 * std, 0, 255).astype(int).tolist()
            )

    def detect_path_contours(self, minimap_img: np.ndarray) -> List[np.ndarray]:
        """增强版路径检测，结合多种特征"""
        # 1. 多尺度预处理
        gray = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. 多特征融合检测
        color_mask = self._get_path_color_mask(minimap_img)
        edge_mask = self._get_path_edge_mask(blurred)
        combined_mask = cv2.bitwise_or(color_mask, edge_mask)

        # 3. 高级形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4. 基于几何特征的路径筛选
        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        for cnt in contours:
            if self._is_valid_path_contour(cnt):
                valid_contours.append(cnt)

        # 5. 路径聚类 (合并相近路径)
        if len(valid_contours) > 1:
            valid_contours = self._cluster_path_contours(valid_contours)

        return valid_contours

    def _get_path_color_mask(self, img: np.ndarray) -> np.ndarray:
        """自适应颜色检测"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 使用动态调整的颜色范围
        if 'path' in self.color_ranges:
            lower, upper = self.color_ranges['path']
        else:
            lower, upper = self.color_adaptation['initial_color_range']

        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # 去除小噪点
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _get_path_edge_mask(self, gray_img: np.ndarray) -> np.ndarray:
        """基于边缘特征的路径检测"""
        # 自适应Canny边缘检测
        v = np.median(gray_img)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(gray_img, lower, upper)

        # 边缘连接处理
        return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    def _is_valid_path_contour(self, contour: np.ndarray) -> bool:
        """验证轮廓是否符合路径特征"""
        # 基于面积筛选
        area = cv2.contourArea(contour)
        if area < 50:  # 最小面积阈值
            return False

        # 基于几何特征筛选
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height)

        return (aspect_ratio > self.geometry_params['aspect_ratio_thresh'] and
                min(width, height) > self.geometry_params['min_path_width'] and
                max(width, height) > self.geometry_params['min_path_length'])

    def _cluster_path_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """使用聚类算法合并相近路径"""
        # 提取轮廓中心点
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                centers.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])

        if len(centers) < 2:
            return contours

        # DBSCAN聚类
        clustering = DBSCAN(eps=30, min_samples=1).fit(centers)

        # 合并同一簇的轮廓
        clustered_contours = []
        for label in set(clustering.labels_):
            cluster_indices = np.where(clustering.labels_ == label)[0]
            if len(cluster_indices) > 1:
                # 合并轮廓
                combined = np.vstack([contours[i] for i in cluster_indices])
                hull = cv2.convexHull(combined)
                clustered_contours.append(hull)
            else:
                clustered_contours.append(contours[cluster_indices[0]])

        return clustered_contours

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
        # 小地图处理器
        # self.minimap_processor = EnhancedMinimapProcessor(config)
        #
        # self.terrain_memory = TerrainMemory()
        #
        # self.geometryAnalyzer = GeometryAnalyzer(config)

        self.minimap_processor = EnhancedMinimapProcessor(config)
        self.terrain_memory = EnhancedTerrainMemory(config)
        self.geometryAnalyzer = GeometryAnalyzer(config)
        # 路径跟踪状态
        self.path_memory = {
            'last_positions': deque(maxlen=20),
            'path_confidence': 0,
            'lost_counter': 0
        }

        # 视觉导航参数
        self.visual_nav_params = {
            'max_lost_frames': 30,  # 连续丢失路径的最大帧数
            'min_path_confidence': 0.7,  # 路径可信度阈值
            'exploration_radius': 200  # 探索半径(像素)
        }

    def _detect_path(self, frame):
        """基于视觉特征检测无标记路径"""
        # 1. 提取小地图区域 (右上角)
        h, w = frame.shape[:2]
        minimap = frame[:h // 4, w - w // 4:]  # 假设小地图在右上角1/4区域

        # 2. 使用颜色和纹理特征检测路径
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 3. 形态学处理增强路径连续性
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # 4. 寻找最大连通区域作为主路径
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest_cnt)
        return None

    def execute(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """增强版导航主循环"""
        # 1. 更新小地图信息
        minimap = self._extract_minimap(frame)
        self.minimap_processor.update(minimap)

        # 2. 多模式路径检测
        navigation_result = None
        if self.path_memory['path_confidence'] > self.visual_nav_params['min_path_confidence']:
            navigation_result = self._follow_memorized_path(frame)

        if not navigation_result:
            navigation_result = self._navigate_by_minimap(minimap)

        if not navigation_result:
            navigation_result = self._navigate_by_visual(frame)

        if not navigation_result:
            navigation_result = self._explore_safely(frame)

        # 3. 更新导航状态
        self._update_navigation_state(navigation_result)

    def _extract_minimap(self, frame: np.ndarray) -> np.ndarray:
        """提取小地图区域"""
        h, w = frame.shape[:2]
        return frame[:h // 4, w - w // 4:]  # 假设小地图在右上角

    def _follow_memorized_path(self, frame: np.ndarray, predicted_direction=None) -> Optional[Dict]:
        """跟随记忆路径"""
        # 实现基于记忆的路径跟踪
        if len(self.path_memory['last_positions']) < 5:
            return None

        # 计算预测位置
        # ... (实现路径预测逻辑)

        return {
            'type': 'memorized',
            'direction': predicted_direction,
            'confidence': self.path_memory['path_confidence']
        }

    def _navigate_by_minimap(self, minimap: np.ndarray) -> Optional[Dict]:
        """基于小地图导航"""
        contours = self.minimap_processor.detect_path_contours(minimap)
        if not contours:
            return None

        # 选择最优路径
        main_path = max(contours, key=cv2.contourArea)
        direction = self._calculate_direction_from_contour(main_path)

        # 更新颜色特征
        x, y, w, h = cv2.boundingRect(main_path)
        path_region = minimap[y:y + h, x:x + w]
        self.minimap_processor.update_color_profile(path_region)

        return {
            'type': 'minimap',
            'direction': direction,
            'confidence': 0.9  # 小地图通常高可信度
        }

    def _navigate_by_visual(self, frame: np.ndarray) -> Optional[Dict]:
        """基于主画面视觉导航"""
        # 1. 检测可行走区域
        walkable_mask = self._get_walkable_area(frame)

        # 2. 寻找最佳路径方向
        direction = self._find_best_direction(walkable_mask)
        if direction is None:
            return None

        return {
            'type': 'visual',
            'direction': direction,
            'confidence': 0.7
        }

    def _explore_safely(self, frame: np.ndarray) -> Dict:
        """安全探索模式"""
        # 结合3D感知和地形记忆
        safe_direction = self.terrain_memory.get_safe_direction()
        if safe_direction is None:
            safe_direction = self._get_random_exploration_direction()

        return {
            'type': 'exploration',
            'direction': safe_direction,
            'confidence': 0.5
        }

    def _get_walkable_area(self, frame: np.ndarray) -> np.ndarray:
        """获取可行走区域掩码"""
        # 1. 深度估计
        depth_map = self.depth_estimator.estimate(frame)

        # 2. 几何分析
        walkable_mask = self.geometryAnalyzer.get_walkable_mask(depth_map)

        # 3. 结合语义信息
        detections = self.detector.detect(frame)
        for det in detections:
            if det['class_name'] in self.config['navigation']['obstacle_classes']:
                x1, y1, x2, y2 = map(int, det['bbox'])
                walkable_mask[y1:y2, x1:x2] = 0

        return walkable_mask

    def _find_best_direction(self, walkable_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """在可行走区域中寻找最佳前进方向"""
        # 实现基于光流或路径搜索的方向检测
        # ... (具体实现)
        return (1.0, 0.0)  # 示例返回值

    def _update_navigation_state(self, result: Dict) -> None:
        """更新导航状态机"""
        if result['confidence'] > self.visual_nav_params['min_path_confidence']:
            self.path_memory['path_confidence'] = result['confidence']
            self.path_memory['lost_counter'] = 0
        else:
            self.path_memory['lost_counter'] += 1

        # 如果长时间丢失路径，重置状态
        if self.path_memory['lost_counter'] > self.visual_nav_params['max_lost_frames']:
            self.path_memory['path_confidence'] = 0
            self.path_memory['last_positions'].clear()


class EnhancedTerrainMemory(TerrainMemory):
    """增强版地形记忆系统"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.safety_map = None  # 安全区域地图
        self.exploration_map = None  # 探索热度图

    def update(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """增强版地形更新"""
        # 1. 更新障碍物信息
        super().update([d for d in detections if d['class_name'] in self.config['navigation']['obstacle_classes']])

        # 2. 更新安全区域地图
        self._update_safety_map(frame)

        # 3. 更新探索热度图
        self._update_exploration_map(frame)

    def get_safe_direction(self) -> Optional[Tuple[float, float]]:
        """获取安全探索方向"""
        # 实现基于安全区域和探索热度的方向计算
        # ... (具体实现)
        return None

    def _update_safety_map(self, frame: np.ndarray) -> None:
        """更新安全区域地图"""
        # 实现安全区域检测
        pass

    def _update_exploration_map(self, frame: np.ndarray) -> None:
        """更新探索热度图"""
        # 实现探索区域记录
        pass


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