import os, cv2
import albumentations as A
from tqdm import tqdm

# 路径自动定位
DATA_DIR = "../dataset/train"
CLASSES = ['friction', 'imbalance', 'loose', 'normal']
TARGET_PER_CLASS = 2000 

# 适配最新版 Albumentations 的参数
transform = A.Compose([
    A.Resize(height=224, width=224),
    # 修正 GaussNoise 参数
    A.GaussNoise(std_range=(0.04, 0.2), p=0.7), 
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=10, p=0.5),
    # 修正 CoarseDropout 参数
    A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3)
])

for cls in CLASSES:
    path = os.path.join(DATA_DIR, cls)
    
    # 检查文件夹是否存在，不存在则跳过并提示
    if not os.path.exists(path):
        print(f"⚠️ 警告：找不到目录 {path}，请确保原始图片已放入该文件夹。")
        continue
        
    originals = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
    if len(originals) == 0:
        print(f"⚠️ 警告：文件夹 {cls} 是空的，请放入至少 1 张原始图片。")
        continue

    print(f"正在工业化扩增类别: {cls}...")
    count = len(originals)
    pbar = tqdm(total=TARGET_PER_CLASS - count)
    
    idx = 0
    while count < TARGET_PER_CLASS:
        img_name = originals[idx % len(originals)]
        image = cv2.imread(os.path.join(path, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 执行增强
        aug_img = transform(image=image)['image']
        save_path = os.path.join(path, f"aug_v6_{count}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        
        count += 1
        idx += 1
        pbar.update(1)
    pbar.close()