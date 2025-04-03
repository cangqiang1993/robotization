""""障碍物检测验证"""
import cv2




def test_on_recorded_frames():
    detector = ObstacleDetector(config)
    visualizer = DebugVisualizer()

    for frame in sim.get_frame():
        obstacles = detector.detect(frame)
        visualizer.display(frame, obstacles)

        # 手动验证或与标注比对
        if cv2.waitKey(100) == ord('q'):
            break

# 输出结果示例：
# [frame_0012.jpg] 检测到障碍物: 岩石(置信度0.92) 位置(x=320,y=180)
# [frame_0013.jpg] 漏检: 左侧毒沼未识别