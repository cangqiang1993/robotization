from collections import deque

import cv2
import time

import psutil
import pyautogui
import torch
import yaml
from pathlib import Path

from numpy import random
from pynput import keyboard
from yolov5.utils.loggers.comet import config

from src.mmo.combat_3d import MMO_CombatSystem
from src.mmo.navigation_3d import MMO_Navigator
from src.mmo.perception_3d import ThreeDPerception
from src.monitor import GameMonitor
from .detector import YOLODetector
from .obs_capture import OBSCapture
from .navigation import Navigator
from .combat import CombatSystem


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
    def __init__(self, config_path='config/settings.yaml'):
        # 加载配置
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # 初始化模块
        self.detector = YOLODetector(
            weights_path=self.config['yolo']['weights_path'],
            device=self.config['yolo']['device'])

        self.navigator = Navigator(self.config, self.detector)
        self.combat_system = CombatSystem(self.config, self.detector)

        # 3D MMO专用初始化
        if self.config['game']['type'] == '3dmmo':
            self.perception_3d = ThreeDPerception(self.config)
            self.navigator = MMO_Navigator(self.config, self.detector)
            self.combat_system = MMO_CombatSystem(self.config, self.detector)
        else:
            self.navigator = Navigator(self.config, self.detector)
            self.combat_system = CombatSystem(self.config, self.detector)

        # 反检测模块
        self.antidetection = AntiDetection(self.config.get('antidetection', {}))

        # OBS捕获
        self.obs = OBSCapture(self.config['obs']['camera_index'])

        # 状态控制
        self.running = False
        self.paused = False
        self.current_mode = 'explore'  # explore/combat/follow
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
        """判断是否应该进入战斗状态"""
        monster_detections = [d for d in detections if d['class_name'] in self.config['combat']['monster_classes']]
        return len(monster_detections) >= self.config['combat']['min_monsters']

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


if __name__ == "__main__":
    bot = GameBot()
    bot.start()