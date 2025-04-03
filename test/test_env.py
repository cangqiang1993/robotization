from pathlib import Path

import cv2

"""构建模拟测试环境"""
class VideoSimulator:
    def __init__(self, frame_dir):
        self.frames = sorted(Path(frame_dir).glob("*.jpg"))
        self.current_idx = 0

    def get_frame(self):
        if self.current_idx >= len(self.frames):
            return None
        frame = cv2.imread(str(self.frames[self.current_idx]))
        self.current_idx += 1
        return frame


# 初始化模拟器
sim = VideoSimulator("frames/boss_room")