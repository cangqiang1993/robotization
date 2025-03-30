import math

import numpy as np
import pyautogui
import time
import random

from src.utils import extract_roi, find_color_mask, ColorRange


class CombatSystem:
    def __init__(self, config, detector):
        self.config = config
        self.detector = detector
        self.skill_keys = config['combat']['skill_keys']
        self.potion_key = config['combat']['potion_key']
        self.retreat_threshold = config['combat']['retreat_threshold']
        self.min_health = 0.3  # 30%血量以下强制撤退

    def should_retreat(self, frame):
        """综合判断是否需要撤退"""
        return (self._check_health(frame) < self.min_health or
                self._check_enemy_density(frame) > 5 or
                time.time() - self.combat_start_time > 120)  # 战斗超时

    def _check_health(self, frame):
        """检测血量百分比"""
        hp_region = extract_roi(frame, self.config['ui']['hp_bar_region'])
        hp_mask = find_color_mask(hp_region, ColorRange.RED)
        return np.count_nonzero(hp_mask) / (hp_mask.shape[0] * hp_mask.shape[1])

    def _check_enemy_density(self, frame):
        """检测周围敌人密度"""
        detections = self.detector.detect(frame)
        enemies = [d for d in detections if d['class_name'] in self.config['combat']['monster_classes']]
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        return sum(1 for e in enemies if
                   math.hypot(e['bbox'][0] - center[0], e['bbox'][1] - center[1]) < 300)

    def execute(self, frame, detections):
        """执行战斗逻辑"""
        monsters = self.select_targets(detections)
        start_time = time.time()

        while time.time() - start_time < self.config['combat']['max_combat_duration']:
            # 优先攻击最近的怪物
            target = self.select_primary_target(monsters, frame.shape)
            if not target:
                break

            self.attack_target(target)

            # 检查是否需要使用药水
            if random.random() < self.config['combat']['potion_probability']:
                self.use_potion()

            # 更新检测结果
            detections = self.detector.detect(frame)
            monsters = self.select_targets(detections)

    def select_targets(self, detections):
        """筛选有效的怪物目标"""
        return [d for d in detections
                if d['class_name'] in self.config['combat']['monster_classes']
                and d['confidence'] > self.config['combat']['min_confidence']]

    def select_primary_target(self, monsters, frame_shape):
        """选择最近的怪物作为主要目标"""
        if not monsters:
            return None

        # 计算屏幕中心
        center_x, center_y = frame_shape[1] // 2, frame_shape[0] // 2

        # 找出距离中心最近的怪物
        def distance_to_center(monster):
            x1, y1, x2, y2 = monster['bbox']
            monster_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            return ((monster_center[0] - center_x) ** 2 +
                    (monster_center[1] - center_y) ** 2) ** 0.5

        return min(monsters, key=distance_to_center)

    def attack_target(self, target):
        """攻击指定目标"""
        x1, y1, x2, y2 = target['bbox']
        target_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # 移动鼠标到目标位置
        pyautogui.moveTo(target_center[0], target_center[1],
                         duration=random.uniform(0.1, 0.3))

        # 使用技能组合
        for skill in random.sample(self.skill_keys,
                                   random.randint(1, len(self.skill_keys))):
            pyautogui.press(skill)
            time.sleep(random.uniform(0.1, 0.3))

        # 普通攻击
        pyautogui.click(button='left')
        time.sleep(random.uniform(0.2, 0.5))

    def use_potion(self):
        """使用药水"""
        pyautogui.press(self.potion_key)
        time.sleep(0.5)