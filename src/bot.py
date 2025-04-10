import logging
from collections import deque

import cv2
import time

import numpy as np
import psutil
import pyautogui
import yaml
from pathlib import Path

from numpy import random
from pynput import keyboard

from src.images_array import ImagesPathfinding
from src.mmo.combat_3d import MMO_CombatSystem
from src.mmo.navigation_3d import MMO_Navigator
from src.mmo.perception_3d import ThreeDPerception
from src.monitor import GameMonitor
from src.videoFrameLoader import VideoFrameLoader
from .detector import YOLODetector
from .obs_capture import OBSCapture
from .navigation import Navigator
from .combat import CombatSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("路径检测--1")


def load_config(config_path: str = None) -> dict:
    """加载YAML配置文件"""
    if config_path is None:
        config_path = Path("settings.yaml")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path.absolute()}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_fields = {
        "obs": ["mode"],
        "performance": ["target_fps"]
    }
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"配置缺少必要部分: {section}")
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"配置缺少必要字段: {section}.{field}")

    return config



class AntiDetection:
    """反检测系统，模拟人类操作行为"""

    def __init__(self, config):
        self.config = config
        self.action_history = deque(maxlen=100)  # 操作历史记录
        self.last_action_time = time.time()

        # 从配置加载参数
        self.delays = self.config.get('delays', {
            'move': 0.15,
            'skill': 0.3,
            'turn': 0.25
        })

    def random_delay(self, action_type):
        """为操作添加随机延迟"""
        base_delay = self.delays.get(action_type, 0.2)
        delay = random.uniform(base_delay * 0.7, base_delay * 1.3)  # ±30%波动
        time.sleep(max(0.05, delay))  # 最小延迟50ms

    def humanize_mouse(self, dx, dy):
        """人类化鼠标移动轨迹"""
        steps = max(5, int((dx ** 2 + dy ** 2) ** 0.5 / 10))  # 根据距离计算步数
        for i in range(steps):
            progress = i / steps
            x = int(dx * progress)
            y = int(dy * progress)
            pyautogui.moveRel(x, y, duration=0.01)
            time.sleep(random.uniform(0.01, 0.03))

    def random_break(self):
        """随机休息行为"""
        if random.random() < 0.005:  # 0.5%概率触发
            duration = random.uniform(1.0, 5.0)  # 1-5秒随机休息
            time.sleep(duration)


class GameBot:
    def __init__(self, config_path, walk_circuit_path):
        self.config = load_config(config_path)
        self.imgs = ImagesPathfinding(walk_circuit_path)
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化模块
        result = self.imgs.load_images()
        # 处理返回结果（单图或列表）
        if isinstance(result, list):
            print(f"加载了 {len(result)} 张图像")
            display_img = result[0]  # 默认显示第一张
        else:
            print("加载了单张图像")
            display_img = result

        # 显示图像（确保是numpy数组）
        if isinstance(display_img, np.ndarray):
            cv2.imshow("Loaded Image", display_img)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            raise ValueError("图像格式错误，无法显示")
        self.detector = YOLODetector(self.config)
        # 统一处理输入（单图或列表都可用）
        self.detector.detect(result if isinstance(result, list) else [result])
        # 3D MMO专用初始化
        if self.config['game']['type'] == '3dmmo':
            self.perception_3d = ThreeDPerception(self.config)
            self.navigator = MMO_Navigator(self.config, self.detector)
            self.combat_system = MMO_CombatSystem(self.config, self.detector)
        else:
            self.navigator = Navigator(self.config, self.detector)
            self.combat_system = CombatSystem(self.config, self.detector)

        self.load_video_frames(walk_circuit_path)  # 加载视频帧和构建地图（需在 perception_3d 初始化后调用）
        self.plan_route()
        self.navigator = Navigator(self.config, self.detector)
        self.combat_system = CombatSystem(self.config, self.detector)

        # 反检测模块
        self.antidetection = AntiDetection(self.config.get('antidetection', {}))

        # OBS捕获
        self.obs = OBSCapture(self.config)

        # 状态控制
        self.running = False
        self.paused = False
        self.current_mode = 'explore'  # explore/combat/follow
        config = load_config("F:/PythonProject/ZhuXIanShiJie/game_auto/config/settings.yaml")
        self.monitor = GameMonitor(config)

    def start(self, detect_start=None):
        """启动机器人"""
        print("Starting YOLOv5 game bot...")
        self.running = True

        # 键盘监听
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()

        try:
            with self.obs:
                fps_counter = 0
                fps = 0
                last_time = time.time()

                while self.running:
                    frame = self.obs.get_frame()

                    # 3D环境感知
                    if hasattr(self, 'perception_3d'):
                        self.perception_3d.update(frame)

                    # 添加人类化延迟
                    self.antidetection.random_delay('frame_process')

                    if self.paused:
                        time.sleep(0.1)
                        continue

                    # 获取游戏画面
                    frame = self.obs.get_frame(self.config['game']['region'])
                    if frame is None:
                        continue

                    # 执行检测
                    detections = self.detector.detect(frame)

                    # 模式决策
                    if self.should_enter_combat(detections):
                        self.current_mode = 'combat'
                        self.combat_system.execute(frame, detections)
                    else:
                        self.current_mode = 'explore'
                        self.navigator.execute(frame, detections)

                    # 计算FPS
                    fps_counter += 1
                    if time.time() - last_time >= 1.0:
                        fps = fps_counter
                        fps_counter = 0
                        last_time = time.time()
                        print(f"FPS: {fps} | Mode: {self.current_mode}")

                    # 更新监控状态
                    self.monitor.update_status(
                        fps=fps,
                        mode=self.current_mode,
                        detection_time=time.time() - detect_start,
                        cpu_usage=psutil.cpu_percent()
                    )

                    if self.config['debug']['show_overlay']:
                        frame = self.monitor.draw_overlay(frame)

        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            self.running = False
            listener.stop()

    def should_enter_combat(self, detections):
        """智能战斗决策"""
        monsters = [d for d in detections if d['class_name'] in self.config['combat']['monster_classes']]

        # 根据怪物密度和路径阻塞情况决定是否战斗
        if not monsters:
            return False

        # 计算怪物阻塞路径的程度
        path_blocked = self._check_path_blockage(monsters)
        min_monsters = self.config['combat'].get('min_monsters', 1)

        return len(monsters) >= min_monsters or path_blocked

    def on_key_press(self, key):
        """键盘监听回调"""
        try:
            if key == keyboard.Key.f1:
                self.running = False
                print("Stopping bot...")
            elif key == keyboard.Key.f2:
                self.paused = not self.paused
                status = "Paused" if self.paused else "Resumed"
                print(f"Bot {status}")
        except AttributeError:
            pass

    # 3D感知
    def build_3d_map(self):
        map_3d = []
        for frame in self.frame_loader.get_frame():
            depth = self.perception_3d.depth_estimator.estimate(frame)
            walkable = self.perception_3d.get_walkable_mask(depth)
            map_3d.append(walkable)
        return np.stack(map_3d)  # 3D体素地图

    def load_video_frames(self, frame_dir):
        """加载预处理好的视频帧序列"""
        self.frame_loader = VideoFrameLoader(frame_dir)  # 使用前文定义的帧加载器
        self.frames = list(self.frame_loader.get_frame())  # 缓存所有帧
        logger.info(f"已加载 {len(self.frames)} 帧副本画面")

    # 基于加载的视频帧，分析3D地形并规划全局最优路径
    def plan_route(self):
        """规划副本全局路径"""
        # 1. 构建3D地图（整合 perception_3d.py 和 pathfinder_3d.py）
        walkable_map = self._build_3d_map()

        # 2. 设置起点（第一帧中心）和终点（最后一帧的BOSS位置）
        start_pos = (0, walkable_map.shape[1] // 2, walkable_map.shape[2] // 2)
        end_pos = (-1, *self._detect_boss_position(self.frames[-1]))

        # 3. 调用A*算法规划路径（使用 pathfinder_3d.py）
        from src.mmo.pathfinder_3d import PathFinder3D
        pathfinder = PathFinder3D(walkable_map)
        return pathfinder.find_path(start_pos, end_pos)

    def _build_3d_map(self):
        """构建3D可行走地图，整合深度感知和几何分析"""
        # 获取第一帧的尺寸作为基准
        base_frame = self.frames[0]
        h, w = base_frame.shape[:2]

        # 初始化3D地图数组
        map_3d = np.zeros((len(self.frames), h, w), dtype=np.uint8)

        for i, frame in enumerate(self.frames):
            # 1. 估计深度图 (确保输出尺寸与输入一致)
            depth_map = self.perception_3d.depth_estimator.estimate(frame)

            # 2. 如果深度图尺寸不匹配，调整到原始帧尺寸
            if depth_map.shape[:2] != (h, w):
                depth_map = cv2.resize(depth_map, (w, h))

            # 3. 计算可行走区域
            walkable_mask = self.perception_3d.get_walkable_mask(depth_map)

            # 4. 确保掩码尺寸正确
            if walkable_mask.shape != (h, w):
                walkable_mask = cv2.resize(walkable_mask.astype(float), (w, h)) > 0.5

            # 5. 将可行走区域存入3D地图
            map_3d[i] = walkable_mask.astype(np.uint8)

            # 6. 可选: 应用形态学操作平滑地图
            kernel = np.ones((3, 3), np.uint8)
            map_3d[i] = cv2.morphologyEx(map_3d[i], cv2.MORPH_CLOSE, kernel)

        # 7. 确保地图连通性 (移除孤立区域)
        for i in range(map_3d.shape[0]):
            _, labels = cv2.connectedComponents(map_3d[i])
            if np.max(labels) > 0:  # 有连通区域
                # 只保留最大的连通区域
                largest_component = np.argmax(np.bincount(labels.flat)[1:]) + 1
                map_3d[i] = (labels == largest_component).astype(np.uint8)

        return map_3d

    def _detect_boss_position(self, frame):
        """检测BOSS位置"""
        # 使用YOLO检测器检测BOSS
        detections = self.detector.detect(frame)

        # 筛选BOSS类别的检测结果
        boss_detections = [
            d for d in detections
            if d['class_name'] in self.config['combat'].get('boss_classes', ['boss'])
        ]

        if not boss_detections:
            # 如果没有检测到BOSS，返回帧中心位置
            h, w = frame.shape[:2]
            return (w // 2, h // 2)

        # 返回第一个BOSS检测框的中心位置
        boss = boss_detections[0]
        x1, y1, x2, y2 = boss['bbox']
        return ((x1 + x2) // 2, (y1 + y2) // 2)


if __name__ == "__main__":
    bot = GameBot()
    bot.start()
