import numpy as np
from heapq import heappush, heappop

class PathFinder3D:
    def __init__(self, walkable_mask):
        self.grid = walkable_mask.astype(int)
        self.directions = [
            (1, 0, 0), (-1, 0, 0),  # x 轴移动
            (0, 1, 0), (0, -1, 0),   # y 轴移动
            (0, 0, 1), (0, 0, -1),    # z 轴移动
            (1, 1, 0), (-1, -1, 0),   # 对角线移动（可选）
        ]

    def _distance(self, pos1, pos2):
        """欧几里得距离"""
        return ((pos1[0] - pos2[0]) ** 2 +
               (pos1[1] - pos2[1]) ** 2 +
               (pos1[2] - pos2[2]) ** 2) ** 0.5

    def _is_valid(self, pos):
        """检查位置是否可行走（3D 网格）"""
        x, y, z = pos
        return (
            0 <= x < self.grid.shape[0] and
            0 <= y < self.grid.shape[1] and
            0 <= z < self.grid.shape[2] and
            self.grid[x, y, z] == 1  # 1 表示可行走
        )

    def find_path(self, start, end):
        """A* 路径搜索"""
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heappop(open_set)

            if self._distance(current, end) < 2:  # 接近终点
                return self._reconstruct_path(came_from, current)

            for dx, dy, dz in self.directions:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                if not self._is_valid(neighbor):
                    continue

                tentative_g = g_score[current] + self._cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heappush(open_set, (f_score, neighbor))
        return None  # 无路径

    def _reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # 反转路径

    def _cost(self, current, neighbor):
        """移动成本（默认 1）"""
        return 1

    def _heuristic(self, neighbor, end):
        """启发式函数（曼哈顿距离）"""
        return (abs(neighbor[0] - end[0]) +
                abs(neighbor[1] - end[1]) +
                abs(neighbor[2] - end[2]))