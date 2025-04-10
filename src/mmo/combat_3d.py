import logging

from src.combat import CombatSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OBS Capture")

class MMO_CombatSystem(CombatSystem):
    def execute(self, frame, detections):
        monsters = self.select_targets(detections)
        if not monsters:
            return

        # 选择优先级目标 (考虑距离、威胁等级等)
        target = self.select_3d_target(monsters, frame)

        # 3D战斗逻辑
        self.position_tracker.update(target)
        self.execute_3d_combat(target, frame)

        # 环境感知
        self.check_surroundings(frame)

    def select_3d_target(self, monsters, frame):
        """3D目标选择策略"""
        scored_targets = []
        for m in monsters:
            score = self._calculate_target_score(m, frame)
            scored_targets.append((score, m))

        return max(scored_targets, key=lambda x: x[0])[1]

    def _calculate_target_score(self, monster, frame):
        """计算目标优先级分数"""
        distance = monster.get('distance', 100)
        threat = self.config['combat']['monster_threat'].get(monster['class_name'], 1)
        target = self.select_3d_target(monster, frame)
        logger.info(f"Selected target: {target['class_name']}, Threat: {self.config['combat']['monster_threat'][target['class_name']]}")  # 分数公式 (可调整权重)
        return threat / (distance + 1)

    def execute_3d_combat(self, target):
        """3D空间战斗策略"""
        # 保持最佳攻击距离
        optimal_range = self.config['combat']['ranges'][target['class_name']]
        current_dist = target['distance']

        if current_dist > optimal_range * 1.2:
            self.approach_target(target)
        elif current_dist < optimal_range * 0.8:
            self.retreat_from_target(target)

        # 考虑Z轴攻击（如跳跃攻击）
        if target['elevation'] > 2:
            self.jump_attack()