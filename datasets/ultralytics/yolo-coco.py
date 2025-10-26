from ultralytics import YOLO
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time

ROOT  = Path(r"C:/Users/I.Kharashun/yolo/coco")        # dataset
YAML  = ROOT / "coco_kpts.yaml"                        # conf Ultralytics
MODEL = "yolo11x-pose.pt"                              # weights

if not YAML.exists():
    YAML.write_text(f"""\
path: {ROOT.as_posix()}
train: images/val2017
val:   images/val2017
test:  images/val2017
kpt_shape: [17, 3]
channels: 3
names:
  0: person
""", encoding="utf-8")

def count_images(img_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in img_dir.glob("*") if p.suffix.lower() in exts)

def main() -> None:
    model = YOLO(MODEL)

    # --- czas end-to-end całego .val() (ocenaу/zapis) ---
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(YAML),
        imgsz=640,
        batch=16,
        workers=0,
        save_json=True,     # predictions.json -> metrics.save_dir
        plots=False
    )
    t1 = time.perf_counter()
    wall_s = t1 - t0

    # --- metryki Ultralytics ---
    p = metrics.pose  # keypoints metrics
    print("\n=== Pose metrics ===")
    print(f"AP@[.5:.95]  = {p.map:.4f}")
    print(f"AP50         = {p.map50:.4f}")
    print(f"AP75         = {p.map75:.4f}")
    print(f"Precision    = {p.mp:.4f}")
    print(f"Recall       = {p.mr:.4f}")

    # --- szypkosc od Ultralytics (ms per image) ---
    sp = getattr(metrics, "speed", {})  # dict: preprocess, inference, postprocess, ...
    pre_ms  = float(sp.get("preprocess", 0.0))
    inf_ms  = float(sp.get("inference", 0.0))
    post_ms = float(sp.get("postprocess", 0.0))

    # FPS srednia
    fps_infer    = 1000.0 / inf_ms if inf_ms > 0 else 0.0
    pipe_ms      = pre_ms + inf_ms + post_ms
    fps_pipeline = 1000.0 / pipe_ms if pipe_ms > 0 else 0.0

    print("\n=== Speed (per image, from Ultralytics) ===")
    print(f"preprocess   : {pre_ms:.2f} ms")
    print(f"inference    : {inf_ms:.2f} ms   ->  {fps_infer:.2f} FPS  (inference-only)")
    print(f"postprocess  : {post_ms:.2f} ms")
    print(f"TOTAL        : {pipe_ms:.2f} ms   ->  {fps_pipeline:.2f} FPS  (pipeline per-image)")

    # --- caly FPS na dataset
    val_dir = ROOT / "images" / "val2017"
    n_images = count_images(val_dir)
    fps_e2e = (n_images / wall_s) if wall_s > 0 else 0.0

    print("\n=== End-to-end throughput (dataset) ===")
    print(f"Images total : {n_images}")
    print(f"Wall time    : {wall_s:.2f} s")
    print(f"E2E FPS      : {fps_e2e:.2f} FPS  (includes evaluation & I/O)")

    # --- COCOeval predictions.json ---
    ann_file = ROOT / "annotations" / "person_keypoints_val2017.json"
    res_file = metrics.save_dir / "predictions.json"

    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(str(res_file))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\n=== COCOeval summary ===")
    coco_eval.summarize()  # AP, AP50, AP75, AP(S/M/L), AR, AR50, …

if __name__ == "__main__":
    main()
