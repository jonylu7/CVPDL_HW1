# CVPDL HW1 - Pig Detection Project Structure

## 📁 Project Directory Layout

```
CVPDL/HW1/
│
├── taica-cvpdl-2025-hw-1/          # Main dataset directory
│   ├── train/                       # Training data
│   │   ├── img/                     # Training images
│   │   │   ├── 00000001.jpg         # Image files (8-digit format)
│   │   │   ├── 00000002.jpg
│   │   │   └── ...                  # Total: 1,266 images
│   │   └── gt.txt                   # Ground truth annotations
│   │
│   ├── test/                        # Test data
│   │   └── img/                     # Test images
│   │       ├── 00000001.jpg         # Image files (8-digit format)
│   │       ├── 00000002.jpg
│   │       └── ...                  # Total: 1,864 images
│   │
│   └── sample_submission.csv        # Sample submission format
│
├── sample_code.py                   # Faster R-CNN training code
├── visualize_bbox.py                # Bounding box visualization tool
├── HW1-Object_Detection_slides.pdf  # Assignment slides
└── PROJECT_STRUCTURE.md             # This file
```

## 📄 File Format Specifications

### 1. Ground Truth File (gt.txt)
**Location:** `taica-cvpdl-2025-hw-1/train/gt.txt`

**Format:** CSV format with 5 columns
```
image_id, x, y, width, height
```

**Example:**
```
1,307,50,96,18
1,308,63,101,41
2,100,200,50,60
```

**Details:**
- **image_id**: Image identifier (e.g., 1 → 00000001.jpg)
- **x**: Bounding box top-left x-coordinate (pixels)
- **y**: Bounding box top-left y-coordinate (pixels)
- **width**: Bounding box width (pixels)
- **height**: Bounding box height (pixels)
- One image can have multiple bounding boxes (multiple rows with same image_id)
- Total annotations: 38,748 rows

### 2. Image Files
**Format:** JPEG images with 8-digit zero-padded filenames

**Naming Convention:**
- `00000001.jpg` (image_id = 1)
- `00000002.jpg` (image_id = 2)
- `00001234.jpg` (image_id = 1234)

**Training Set:**
- Location: `taica-cvpdl-2025-hw-1/train/img/`
- Total images: 1,266

**Test Set:**
- Location: `taica-cvpdl-2025-hw-1/test/img/`
- Total images: 1,864

### 3. Submission File Format
**Example:** `sample_submission.csv`

**Format:**
```csv
Image_ID,PredictionString
1,0.95 100.5 200.3 50.0 60.0 0 0.89 150.0 210.0 55.0 65.0 0
2,0.87 80.0 120.0 45.0 55.0 0
```

**Details:**
- **Image_ID**: Test image identifier (integer)
- **PredictionString**: Space-separated detections
  - Each detection: `confidence x y w h 0`
  - confidence: Detection confidence score (0-1)
  - x, y: Top-left coordinates
  - w, h: Width and height
  - 0: Class label (always 0 for pig)
  - Multiple detections separated by spaces

## 🔧 Available Tools

### Training Scripts
- **sample_code.py**: Faster R-CNN training implementation
  - Uses PyTorch and torchvision
  - Pre-trained on COCO dataset
  - Includes train/val split (80/20)
  - Saves model weights

### Visualization Tools
- **visualize_bbox.py**: Interactive bounding box visualization
  - Load and display ground truth annotations
  - Support for single or multiple images
  - Statistical analysis of dataset

## 📊 Dataset Statistics

- **Training Images**: 1,266
- **Test Images**: 1,864
- **Total Annotations**: 38,748
- **Average boxes per image**: ~30.6
- **Classes**: 1 (pig)
- **Image Format**: JPG/JPEG
- **Annotation Format**: CSV (x, y, w, h)

## 🚀 Quick Start

1. **Visualize annotations:**
   ```bash
   python visualize_bbox.py
   ```

2. **Train model:**
   ```bash
   python sample_code.py
   ```

3. **Output:** Generates `submission.csv` for competition

## 📝 Notes

- All image filenames use 8-digit zero-padding
- Bounding boxes are in absolute pixel coordinates
- Multiple pigs can appear in a single image
- Test set has no ground truth annotations

