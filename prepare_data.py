import os
import cv2

imgs_path = r'data\train\images'
annotations_path = r'data\train\masks'

def prepare_data():
  
  for img_path in os.listdir(imgs_path):
    
    if img_path[-4:] != '.png':
      img = cv2.imread(f'{imgs_path}\{img_path}')
      cv2.imwrite(rf'cars\Car parts dataset\File1\img\{img_path[:-4]}.png', img)
      os.remove(f'{imgs_path}\{img_path}')

def resize():
  
  for img_name in os.listdir(imgs_path):
    img_path = os.path.join(imgs_path, img_name)
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (400, 300))
    cv2.imwrite(img_path, img_resized)

  for ann_name in os.listdir(annotations_path):
    ann_path = os.path.join(annotations_path, ann_name)
    ann = cv2.imread(ann_path)
    ann_resized = cv2.resize(ann, (400, 300))
    cv2.imwrite(ann_path, ann_resized)

if __name__ == '__main__':
  # prepare_data()
  resize()