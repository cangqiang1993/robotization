# 副本流程状态机
class DungeonStateMachine:
    STATES = ['moving', 'fighting', 'looting', 'boss']

    def __init__(self):
        self.current_state = 'moving'

    def update(self, game_state):
        if self.current_state == 'moving' and game_state['enemies_nearby']:
            self.transition_to('fighting')
        # 其他状态转换逻辑...