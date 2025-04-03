# test_pathfinding.py
""""路径规划算法测试"""
import cv2



def test_navigation_sequence():
    nav = Navigator3D(config)
    waypoints = [(100, 100), (400, 300)]  # 预设路径点

    for frame in sim.get_frame():
        depth = load_depth_map(frame)  # 从配套深度图加载
        walkable = nav.generate_walkable_map(depth)
        path = nav.plan_path(waypoints, walkable)

        # 可视化结果
        cv2.polylines(frame, [path], False, (0, 255, 0), 2)
        cv2.imshow("Path Planning", frame)
        cv2.waitKey(50)