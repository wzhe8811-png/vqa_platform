import json
import os
from collections import Counter

# 加载问题文件
questions_file = './data/test_subset/questions/testdev_balanced_questions.json'
with open(questions_file, 'r', encoding='utf-8') as f:
    raw_questions_data = json.load(f)

# 获取所有问题项
if isinstance(raw_questions_data, dict):
    all_items = raw_questions_data.values()
elif isinstance(raw_questions_data, list):
    all_items = raw_questions_data
else:
    raise ValueError(f"Unexpected data format: {type(raw_questions_data)}")

# 收集所有 imageId
all_image_ids = []
for item in all_items:
    # 根据我们之前的发现，图片ID可能是 'imageId' 或 'image'
    image_id = item.get('imageId') or item.get('image')
    if image_id:
        all_image_ids.append(image_id)
    else:
        print(f"Warning: Item missing 'imageId' or 'image' field. Keys: {list(item.keys())}")

print(f"Total unique questions: {len(all_items)}")
print(f"Total unique image IDs found: {len(set(all_image_ids))}")

# 获取图片目录中的所有文件名（去掉扩展名）
image_dir = './data/test_subset/images/'
existing_images_no_ext = set()
for filename in os.listdir(image_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
        existing_images_no_ext.add(name)

print(f"Total unique images in dir: {len(existing_images_no_ext)}")

# 分析匹配情况
matched_count = 0
missing_images = []
for image_id in set(all_image_ids):
    if image_id in existing_images_no_ext:
        matched_count += 1
    else:
        missing_images.append(image_id)

print(f"\nMatched images: {matched_count}")
print(f"Missing images: {len(missing_images)}")

if missing_images:
    print("\nFirst 10 missing image IDs:")
    for img_id in sorted(missing_images)[:10]:
        print(f"  - {img_id}")

# 如果所有图片都在，我们还需要修改主脚本以添加扩展名
if len(missing_images) == 0:
    print("\nAll images seem to exist. The script needs to be modified to append extension to image path.")
else:
    print(f"\nThere are missing images. We will process only those that exist.")

# 2. 创建一个新版本的 run_tot_on_gqa.py，它能正确处理图片扩展名
print("\nCreating updated script: run_tot_on_gqa_with_images.py ...")

with open('run_tot_on_gqa.py', 'r', encoding='utf-8') as f:
    main_script_content = f.read()

# 查找并修改构建 image_path 的部分
# 我们需要找到这一行，并在其后添加查找扩展名的逻辑
search_str = "image_path = os.path.join(args.input_dir, 'images', image_filename)"
replace_str = '''        # Find the actual image file with extension
        image_base_name = image_filename
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        image_path = None
        for ext in image_extensions:
            potential_path = os.path.join(args.input_dir, 'images', image_base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"Warning: Image file not found for base name: {image_base_name}. Skipping this question.")
            continue # Skip this question if no image is found'''

updated_script_content = main_script_content.replace(search_str, replace_str)

# 将新内容写入一个新文件
with open('run_tot_on_gqa_with_images.py', 'w', encoding='utf-8') as f:
    f.write(updated_script_content)

print("New script 'run_tot_on_gqa_with_images.py' created successfully!")

