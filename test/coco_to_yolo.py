import json
import os
from tqdm import tqdm


def coco2yolo_batch(input_dir, output_dir, class_names=None):
    """
    批量将COCO JSON（ID从1开始）转换为YOLO格式（ID从0开始）
    :param input_dir: 包含COCO JSON文件的目录
    :param output_dir: 输出YOLO格式的目录
    :param class_names: 可选，指定类别顺序（若提供则按此顺序重新映射ID）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 定义原始COCO类别（ID从1开始）
    coco_categories = {
        1: "car",
        2: "monster_normal",
        3: "monster_elite",
        4: "monster_boss",
        5: "npc_friendly",
        6: "ui_button",
        7: "item_health",
        8: "item_mana",
        9: "minimap_marker",
        10: "tree",
        11: "rock",
        12: "wall",
        13: "danger_zone",
        14: "safe_path"
    }

    # 生成YOLO格式的classes.txt（ID从0开始）
    yolo_categories = {k - 1: v for k, v in coco_categories.items()}
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for cat_id in sorted(yolo_categories.keys()):
            f.write(f"{yolo_categories[cat_id]}\n")

    # 处理每个JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    for json_file in tqdm(json_files, desc="Converting COCO to YOLO"):
        json_path = os.path.join(input_dir, json_file)

        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # 构建图像ID到文件名的映射
        images = {img['id']: img['file_name'] for img in coco_data['images']}

        # 处理标注
        for img_id, img_info in images.items():
            txt_filename = os.path.splitext(img_info)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)

            with open(txt_path, 'w') as f_txt:
                for ann in coco_data['annotations']:
                    if ann['image_id'] == img_id:
                        # 转换类别ID（COCO 1-14 → YOLO 0-13）
                        coco_id = ann['category_id']
                        yolo_id = coco_id - 1

                        # 如果提供了class_names，则按指定顺序重新映射
                        if class_names:
                            try:
                                cat_name = coco_categories[coco_id]
                                yolo_id = class_names.index(cat_name)
                            except (KeyError, ValueError) as e:
                                print(f"警告：跳过未知类别ID {coco_id} 或名称 {cat_name}")
                                continue

                        # 转换边界框坐标
                        x, y, w, h = ann['bbox']
                        img_width = next(img['width'] for img in coco_data['images'] if img['id'] == img_id)
                        img_height = next(img['height'] for img in coco_data['images'] if img['id'] == img_id)

                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        w_norm = w / img_width
                        h_norm = h / img_height

                        # 写入YOLO格式
                        f_txt.write(f"{yolo_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


if __name__ == '__main__':
    # 配置路径
    input_dir = "C:/Users/35522/Desktop/模型/验证-2/annotations"  # COCO JSON目录
    output_dir = "C:/Users/35522/Desktop/转换YOLO/yolo_labels"  # 输出目录

    # 可选：自定义类别顺序（必须包含所有类别）
    class_names = ["car", "monster_normal", "monster_elite", "monster_boss", "npc_friendly", "ui_button", "item_health",
                   "item_mana", "minimap_marker", "tree", "rock", "wall", "danger_zone", "safe_path"]  # 替换为你的实际类别

    # 运行转换
    coco2yolo_batch(input_dir, output_dir, class_names)
    print("转换完成！结果保存在:", output_dir)