import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

# ------------------------------
# Configuration
# ------------------------------
# Choose path based on your environment
IS_COLAB = False  # Set to True if running in Colab

if IS_COLAB:
    # Colab paths
    TRAIN_IMG_DIR = '/content/drive/My Drive/CVPDL_datasets/train/img'
    TRAIN_GT_FILE = '/content/drive/My Drive/CVPDL_datasets/train/gt.txt'
else:
    # Local paths (modify according to your setup)
    TRAIN_IMG_DIR = 'taica-cvpdl-2025-hw-1/train/img'
    TRAIN_GT_FILE = 'taica-cvpdl-2025-hw-1/train/gt.txt'

# ------------------------------
# Load GT Annotations
# ------------------------------
def load_gt(gt_file):
    """Load ground truth file"""
    boxes_dict = {}
    with open(gt_file) as f:
        for line in f:
            line = line.strip().split(',')
            if len(line) < 5:
                continue
            
            img_id = line[0].strip()
            x, y, w, h = map(float, line[1:5])
            
            # Filter invalid bounding boxes
            if w <= 0 or h <= 0:
                continue
            
            if img_id not in boxes_dict:
                boxes_dict[img_id] = []
            
            # Store in [x, y, w, h] format
            boxes_dict[img_id].append([x, y, w, h])
    
    return boxes_dict

# ------------------------------
# Visualization Functions
# ------------------------------
def visualize_image_with_bbox(img_path, boxes, title="Image with Bounding Boxes"):
    """Draw bounding boxes on image"""
    # Load image
    img = Image.open(img_path)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Draw each bounding box
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    for i, (x, y, w, h) in enumerate(boxes):
        # Create rectangle
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            x, y - 5,
            f'Pig {i+1}',
            color=colors[i % len(colors)],
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    ax.set_title(f"{title}\nTotal boxes: {len(boxes)}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_by_img_id(img_id, boxes_dict, img_dir):
    """Visualize image by ID"""
    # Build image path (filename is 8-digit number, e.g., 00000001.jpg)
    img_name = f"{str(img_id).zfill(8)}.jpg"
    img_path = os.path.join(img_dir, img_name)
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return
    
    # Get bounding boxes
    boxes = boxes_dict.get(img_id, [])
    if len(boxes) == 0:
        print(f"⚠️  No bounding boxes found for image {img_id}")
        return
    
    # Visualize
    visualize_image_with_bbox(img_path, boxes, title=f"Image ID: {img_id}")
    print(f"✅ Displayed image {img_id} with {len(boxes)} bounding box(es)")

def visualize_random_samples(boxes_dict, img_dir, num_samples=5):
    """Randomly display images with annotations"""
    # Get all image IDs with annotations
    img_ids = list(boxes_dict.keys())
    
    if len(img_ids) == 0:
        print("❌ No images with annotations found!")
        return
    
    # Random selection
    num_samples = min(num_samples, len(img_ids))
    sample_ids = random.sample(img_ids, num_samples)
    
    print(f"📊 Visualizing {num_samples} random samples...\n")
    
    for img_id in sample_ids:
        visualize_by_img_id(img_id, boxes_dict, img_dir)
        print("-" * 50)

# ------------------------------
# Main Program
# ------------------------------
if __name__ == "__main__":
    print("🐷 Pig Detection - Bounding Box Visualization\n")
    
    # Check if files exist
    if not os.path.exists(TRAIN_GT_FILE):
        print(f"❌ GT file not found: {TRAIN_GT_FILE}")
        print("Please update the path in the script!")
        exit(1)
    
    if not os.path.exists(TRAIN_IMG_DIR):
        print(f"❌ Image directory not found: {TRAIN_IMG_DIR}")
        print("Please update the path in the script!")
        exit(1)
    
    # Load annotations
    print(f"📖 Loading annotations from: {TRAIN_GT_FILE}")
    boxes_dict = load_gt(TRAIN_GT_FILE)
    print(f"✅ Loaded {len(boxes_dict)} images with annotations\n")
    
    # Statistics
    total_boxes = sum(len(boxes) for boxes in boxes_dict.values())
    avg_boxes = total_boxes / len(boxes_dict) if len(boxes_dict) > 0 else 0
    print(f"📊 Statistics:")
    print(f"   - Total images with annotations: {len(boxes_dict)}")
    print(f"   - Total bounding boxes: {total_boxes}")
    print(f"   - Average boxes per image: {avg_boxes:.2f}\n")
    
    # Display options
    print("=" * 50)
    print("Choose an option:")
    print("1. Visualize specific image by ID")
    print("2. Visualize random samples (5 images)")
    print("3. Visualize first 5 images")
    print("=" * 50)
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == '1':
        img_id = input("Enter image ID: ").strip()
        visualize_by_img_id(img_id, boxes_dict, TRAIN_IMG_DIR)
    
    elif choice == '2':
        num = input("How many samples? (default 5): ").strip()
        num = int(num) if num.isdigit() else 5
        visualize_random_samples(boxes_dict, TRAIN_IMG_DIR, num)
    
    elif choice == '3':
        img_ids = sorted(boxes_dict.keys(), key=lambda x: int(x))[:5]
        for img_id in img_ids:
            visualize_by_img_id(img_id, boxes_dict, TRAIN_IMG_DIR)
            print("-" * 50)
    
    else:
        print("Invalid choice! Showing 5 random samples by default...")
        visualize_random_samples(boxes_dict, TRAIN_IMG_DIR, 5)

