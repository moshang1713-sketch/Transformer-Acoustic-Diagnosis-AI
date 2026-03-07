import os, cv2, shutil, random
import albumentations as A
from tqdm import tqdm
import numpy as np

# --- 路径自动识别 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")
CLASSES = ['friction', 'imbalance', 'loose', 'normal']
TARGET_TRAIN, TARGET_VAL = 2000, 400

transform = A.Compose([
    A.Resize(height=224, width=224),
    A.GaussNoise(std_range=(0.04, 0.2), p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3)
])

def remix_and_augment():
    for cls in CLASSES:
        # 1. 第一步：在内存中缓存所有原始图像 (核心修复)
        seed_images = []
        for split in ['train', 'val']:
            path = os.path.join(BASE_DIR, split, cls)
            if os.path.exists(path):
                files = [os.path.join(path, f) for f in os.listdir(path) 
                         if f.endswith(('.jpg', '.png')) and 'aug' not in f]
                for fpath in files:
                    img = cv2.imread(fpath)
                    if img is not None:
                        seed_images.append(img)
        
        if not seed_images:
            print(f"⚠️ 类别 {cls} 没找到原图，跳过。")
            continue

        # 2. 第二步：读完之后再安全删除旧文件夹
        for split in ['train', 'val']:
            save_dir = os.path.join(BASE_DIR, split, cls)
            shutil.rmtree(save_dir, ignore_errors=True)
            os.makedirs(save_dir, exist_ok=True)

        # 3. 第三步：打乱顺序并分配种子
        random.shuffle(seed_images)
        split_idx = int(len(seed_images) * 0.8)
        seeds = {
            'train': seed_images[:split_idx] if split_idx > 0 else seed_images,
            'val': seed_images[split_idx:] if split_idx < len(seed_images) else seed_images
        }

        # 4. 第四步：使用内存中的图像进行扩增
        for split in ['train', 'val']:
            target_count = TARGET_TRAIN if split == 'train' else TARGET_VAL
            current_seeds = seeds[split]
            save_dir = os.path.join(BASE_DIR, split, cls)
            
            pbar = tqdm(total=target_count, desc=f"🚀 {cls}-{split}")
            for i in range(target_count):
                # 从内存中取图，不再受硬盘文件删除的影响
                image = current_seeds[i % len(current_seeds)]
                
                # 转换颜色并增强
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                aug_img = transform(image=image_rgb)['image']
                
                # 保存
                cv2.imwrite(os.path.join(save_dir, f"aug_v7_{i}.jpg"), 
                            cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                pbar.update(1)
            pbar.close()

if __name__ == "__main__":
    remix_and_augment()