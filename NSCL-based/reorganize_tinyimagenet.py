import os
import shutil

def reorganize_val(dataroot='./data/tiny-imagenet-200'):
    val_dir = os.path.join(dataroot, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    # Check if already reorganized
    if not os.path.exists(val_images_dir):
        print("Validation folder already reorganized or images folder doesn't exist.")
        return
    
    # Read annotations
    with open(val_annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name = parts[0]
            class_name = parts[1]
            
            # Create class directory if it doesn't exist
            class_dir = os.path.join(val_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Move image to class directory
            src = os.path.join(val_images_dir, img_name)
            dst = os.path.join(class_dir, img_name)
            if os.path.exists(src):
                shutil.move(src, dst)
    
    # Remove empty images directory
    if os.path.exists(val_images_dir) and len(os.listdir(val_images_dir)) == 0:
        os.rmdir(val_images_dir)
    
    print("Validation folder reorganized successfully!")

if __name__ == '__main__':
    reorganize_val()