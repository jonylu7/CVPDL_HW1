# yolov9_pipeline.py
# Usage:
#   python yolov9_pipeline.py --epochs 100 --imgsz 640 --batch 16 --device 0
# Notes:
#   - Assumes project structure exactly as described by you.
#   - Parses train/gt.txt into YOLO label files (one .txt per image).
#   - Trains YOLOv9, runs inference on test/img, and writes submission CSV:
#       Image_ID,PredictionString
#       1,0.95 x y w h 0 0.93 x y w h 0 ...
#   - Coordinates in CSV are PIXELS (left-top x,y,width,height).
#   - Class id is kept as in gt (default 0 if single class).

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "taica-cvpdl-2025-hw-1"
TRAIN_IMG_DIR = DATASET / "train" / "img"
TEST_IMG_DIR = DATASET / "test" / "img"
GT_FILE = DATASET / "train" / "gt.txt"

WORKDIR = ROOT / "yolov9_workdir"
LABELS_DIR = DATASET / "train" / "labels"  # YOLO labels will be written here
YAML_PATH = ROOT / "yolov9_data.yaml"
REPO_DIR = ROOT / "yolov9"
WEIGHTS_OUT_DIR = ROOT / "weights"
SUBMISSION_DIR = ROOT / "submission"

# -------------- helpers --------------

def run(cmd: List[str], cwd: Path = None):
    print(f"▶️  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.glob("*.jpg")] + [p for p in img_dir.glob("*.png")])

def im_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return w, h

def to_int_id(name_or_id: str) -> int:
    """Convert '00000001.jpg' -> 1 ; '1' -> 1"""
    s = name_or_id.strip()
    s = s.replace(".jpg", "").replace(".png", "")
    s = re.sub(r"^0+","", s)  # strip leading zeros
    return int(s) if s else 0

# -------------- GT parser -> YOLO labels --------------

def parse_gt_line(line: str) -> Tuple[str, List[Tuple[float,float,float,float,int]]]:
    """
    Accepts formats like:
      00000001.jpg,x,y,w,h,cls, x,y,w,h,cls, ...
      00000001  x y w h cls  x y w h cls ...
      1  x y w h cls ...
    Returns (image_name_or_id, [(x,y,w,h,cls), ...]) with pixel coords, top-left x,y.
    """
    line = line.strip()
    if not line:
        return "", []
    # split by comma or space
    toks = re.split(r"[,\s]+", line)
    # first token could be filename or id
    head = toks[0]
    rest = toks[1:]
    # group the rest by 5
    boxes = []
    for i in range(0, len(rest), 5):
        g = rest[i:i+5]
        if len(g) < 5:
            continue
        try:
            x, y, w, h = map(float, g[:4])
            c = int(float(g[4]))
            boxes.append((x, y, w, h, c))
        except Exception:
            continue
    return head, boxes

def convert_gt_to_yolo_labels():
    safe_mkdir(LABELS_DIR)
    imgs = {p.name: p for p in list_images(TRAIN_IMG_DIR)}
    # We’ll accumulate boxes per image
    per_image = {}

    if not GT_FILE.exists():
        raise FileNotFoundError(f"Ground truth file not found: {GT_FILE}")

    with open(GT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            name_or_id, boxes = parse_gt_line(line)
            if not name_or_id:
                continue
            # Map id to filename if id is given
            if name_or_id in imgs:
                img_path = imgs[name_or_id]
                img_name = name_or_id
            else:
                # try zero-padded name
                try:
                    iid = to_int_id(name_or_id)
                    key = f"{iid:08d}.jpg"
                    img_path = imgs.get(key, None)
                    img_name = key if img_path else name_or_id
                except Exception:
                    img_path = imgs.get(name_or_id, None)
                    img_name = name_or_id

            if img_path is None:
                # image not found; skip silently (or warn)
                continue

            W, H = im_size(img_path)
            yolo_lines = per_image.get(img_name, [])

            for (x, y, w, h, c) in boxes:
                # Convert TLWH (pixels) -> YOLO normalized CXCYWH
                cx = (x + w / 2.0) / W
                cy = (y + h / 2.0) / H
                nw = w / W
                nh = h / H
                # clamp
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                nw = min(max(nw, 0.0), 1.0)
                nh = min(max(nh, 0.0), 1.0)
                yolo_lines.append(f"{c} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            per_image[img_name] = yolo_lines

    # write per-image label files
    for img_name, lines in per_image.items():
        lbl_path = LABELS_DIR / f"{Path(img_name).stem}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"✅ Wrote YOLO labels to: {LABELS_DIR}")

# -------------- YAML writer --------------

def write_data_yaml(nc: int, names: List[str] = None):
    names = names or [str(i) for i in range(nc)]
    yaml = f"""# Auto-generated for YOLOv9
train: {TRAIN_IMG_DIR.as_posix()}
val:   {TRAIN_IMG_DIR.as_posix()}  # (簡化) 以 train 充當 val；需要嚴謹評估可自行切分
test:  {TEST_IMG_DIR.as_posix()}

nc: {nc}
names: {names}
"""
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml)
    print(f"✅ Wrote data yaml: {YAML_PATH}")

# -------------- Setup YOLOv9 repo --------------

def ensure_yolov9_repo():
    if not REPO_DIR.exists():
        run(["git", "clone", "https://github.com/WongKinYiu/yolov9.git", str(REPO_DIR)])
    # install requirements (safe to re-run)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=REPO_DIR)

# -------------- Train, Detect, Export --------------

def train_yolov9(args):
    safe_mkdir(WORKDIR)
    proj = (REPO_DIR / "runs" / "train").as_posix()
    name = "hw1_yolov9"
    cmd = [
    "python", "train.py",
        "--data", str(YAML_PATH),
        "--weights", args.weights,
        "--img", str(args.imgsz),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--device", str(args.device),
        "--project", proj,
        "--name", name,
        "--exist-ok",
    ]
    run(cmd, cwd=REPO_DIR)
    # locate best.pt
    # latest exp directory:
    exp_dir = sorted((REPO_DIR / "runs" / "train").glob(f"{name}*"))[-1]
    best_pt = exp_dir / "weights" / "best.pt"
    safe_mkdir(WEIGHTS_OUT_DIR)
    out_pt = WEIGHTS_OUT_DIR / "yolov9_best.pt"
    shutil.copy2(best_pt, out_pt)
    print(f"✅ Copied best weights to: {out_pt}")
    return out_pt

def detect_yolov9(weights_path: Path, args):
    proj = (REPO_DIR / "runs" / "detect").as_posix()
    name = "hw1_yolov9_test"
    cmd = [
        sys.executable, "detect.py",
        "--weights", str(weights_path),
        "--source", str(TEST_IMG_DIR),
        "--img", str(args.imgsz),
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--device", str(args.device),
        "--save-txt", "--save-conf",
        "--project", proj,
        "--name", name, "--exist-ok"
    ]
    run(cmd, cwd=REPO_DIR)
    exp_dir = sorted((REPO_DIR / "runs" / "detect").glob(f"{name}*"))[-1]
    labels_out = exp_dir / "labels"   # per-image .txt with: class conf cx cy w h (normalized)
    print(f"✅ YOLO detect labels: {labels_out}")
    return labels_out

# -------------- Build submission --------------

def load_pred_label_file(p: Path) -> List[Tuple[int, float, float, float, float, float]]:
    """
    YOLO detect labels format per line:
        <class> <conf> <cx> <cy> <w> <h>   (normalized to [0,1])
    Returns list of tuples (cls, conf, cx, cy, w, h)
    """
    preds = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) != 6:
                # sometimes format can be 7 (batch id), try to take last 6
                toks = toks[-6:]
            try:
                cls = int(float(toks[0]))
                conf = float(toks[1])
                cx = float(toks[2]); cy = float(toks[3])
                w = float(toks[4]);  h = float(toks[5])
                preds.append((cls, conf, cx, cy, w, h))
            except Exception:
                continue
    # sort by confidence desc
    preds.sort(key=lambda t: t[1], reverse=True)
    return preds

def normalized_cxcywh_to_xywh_pixels(cx, cy, w, h, W, H):
    px = (cx - w/2.0) * W
    py = (cy - h/2.0) * H
    pw = w * W
    ph = h * H
    return px, py, pw, ph

def make_submission(labels_dir: Path) -> Path:
    safe_mkdir(SUBMISSION_DIR)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_csv = SUBMISSION_DIR / f"submission_{ts}.csv"

    test_images = list_images(TEST_IMG_DIR)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "PredictionString"])
        for img_path in test_images:
            W,H = im_size(img_path)
            stem = img_path.stem  # '00000001'
            image_id = to_int_id(stem)

            lbl_path = labels_dir / f"{stem}.txt"
            pred_str_parts = []
            if lbl_path.exists():
                preds = load_pred_label_file(lbl_path)
                for (cls, conf, cx, cy, w, h) in preds:
                    x, y, ww, hh = normalized_cxcywh_to_xywh_pixels(cx, cy, w, h, W, H)
                    # keep floating format similar to your example: score 6dp, coords 2dp, class int
                    pred_str_parts.extend([
                        f"{conf:.6f}",
                        f"{x:.2f}", f"{y:.2f}", f"{ww:.2f}", f"{hh:.2f}",
                        f"{cls}"
                    ])
            # join with spaces; if no preds, keep empty string or write high-conf dummy if required by leaderboard
            pred_str = " ".join(pred_str_parts)
            writer.writerow([image_id, pred_str])

    print(f"✅ Wrote submission CSV: {out_csv}")
    return out_csv

# -------------- main --------------

def infer_nc_from_gt(default_nc=1) -> int:
    """Roughly infer num classes from gt.txt (max class id + 1)."""
    if not GT_FILE.exists():
        return default_nc
    cls_max = 0
    with open(GT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            _, boxes = parse_gt_line(line)
            for *_, c in boxes:
                cls_max = max(cls_max, c)
    return max(default_nc, cls_max + 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov9-c.pt", help="initial weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="low conf to maximize recall, tune later")
    parser.add_argument("--iou", type=float, default=0.6)
    args = parser.parse_args()

    # sanity checks
    assert TRAIN_IMG_DIR.exists(), f"Missing: {TRAIN_IMG_DIR}"
    assert TEST_IMG_DIR.exists(), f"Missing: {TEST_IMG_DIR}"
    assert GT_FILE.exists(), f"Missing: {GT_FILE}"

    safe_mkdir(WEIGHTS_OUT_DIR)
    convert_gt_to_yolo_labels()
    nc = infer_nc_from_gt(default_nc=1)
    write_data_yaml(nc=nc, names=[str(i) for i in range(nc)])

    ensure_yolov9_repo()
    best_weights = train_yolov9(args)
    labels_dir = detect_yolov9(best_weights, args)
    make_submission(labels_dir)

    print("\n🎉 All done. You’ll find:")
    print(f"  • Best weights  : {WEIGHTS_OUT_DIR / 'yolov9_best.pt'}")
    print(f"  • Submission CSV: {SUBMISSION_DIR} (timestamped)\n")

if __name__ == "__main__":
    main()