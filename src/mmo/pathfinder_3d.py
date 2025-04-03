import numpy as np
from heapq import heappush, heappop
# 3D路径规划

class PathFinder3D:
    def __init__(self, walkable_mask):
        self.grid = walkable_mask.astype(int)
        self.directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (1, 1, 0), (-1, -1, 0)]  # 6方向移动

    def find_path(self, start, end):
        """3D A*路径规划"""
        open_set = []
        heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heappop(open_set)

            if self._distance(current, end) < 2:  # 到达目标
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
        return None  # 无可用路径

    def _is_valid(self, pos):
        """检查位置是否可行走"""
        x, y, z = pos
        return (0 <= x < self.grid.shape[0] and
                0 <= y < self.grid.shape[1] and
                self.grid[x, y] == 1)  # 1表示可行走

    # 生成最优路径
    def plan_route(self):
        walkable_map = self.build_3d_map()
        pathfinder = PathFinder3D(walkable_map)

        # 设置起点(第一帧中心)和终点(最后一帧BOSS位置)
        start = (0, walkable_map.shape[1] // 2, walkable_map.shape[2] // 2)
        end = (-1, *self.detect_boss_position())

        return pathfinder.find_path(start, end)