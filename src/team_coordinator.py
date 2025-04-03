
# 团队协作功能
class TeamCoordinator:
    def coordinate_roles(self):
        """分配团队角色"""
        if self.role == "tank":
            self.hold_aggro()
        elif self.role == "healer":
            self.monitor_team_health()
