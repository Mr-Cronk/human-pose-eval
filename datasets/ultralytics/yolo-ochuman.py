from pathlib import Path
import json
import time
import numpy as np
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(r"C:/Users/I.Kharashun/yolo/ochuman")
IMDIR = ROOT / "images"
ANN   = ROOT / "annotations" / "person_keypoints_val2017.json"
MODEL = "yolo11x-pose.pt"

DEVICE = 'cuda:0'   # 0 / 'cpu'
IMG_SIZE = 640
DO_WARMUP = True    

def main():
    coco_gt = COCO(str(ANN))

    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset:
        coco_gt.dataset['licenses'] = []
    coco_gt.createIndex()

    cat_ids = coco_gt.getCatIds(catNms=['person'])
    person_cat_id = cat_ids[0] if len(cat_ids) else (coco_gt.getCatIds()[0] if coco_gt.getCatIds() else 1)

    model = YOLO(MODEL)

    img_ids = coco_gt.getImgIds()

    # --- WARMUP nie liczymy ---
    if DO_WARMUP and len(img_ids) > 0:
        warm_img_info = coco_gt.loadImgs(img_ids[0])[0]
        warm_img_path = IMDIR / Path(warm_img_info['file_name']).name
        if warm_img_path.exists():
            _ = model.predict(source=str(warm_img_path), imgsz=IMG_SIZE, device=DEVICE, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    results = []
    total_frames = 0
    total_infer_time = 0.0

    t_pipeline_start = time.perf_counter()

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = IMDIR / Path(img_info['file_name']).name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        t0 = time.perf_counter()
        pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_infer_time += (t1 - t0)

        total_frames += 1

        if pred.keypoints is None or len(pred.keypoints) == 0:
            continue

        kps_xy = pred.keypoints.xy
        kps_conf = getattr(pred.keypoints, 'conf', None)
        scores = pred.boxes.conf.cpu().numpy() if pred.boxes is not None else np.zeros((len(kps_xy),), float)

        for i in range(len(kps_xy)):
            xy = kps_xy[i].cpu().numpy()
            if kps_conf is not None:
                vis = (pred.keypoints.conf[i].cpu().numpy() > 0).astype(int) * 2
            else:
                vis = np.full((xy.shape[0],), 2, dtype=int)

            kps = []
            for (x, y), v in zip(xy, vis):
                kps += [float(x), float(y), int(v)]

            results.append({
                "image_id": int(img_id),
                "category_id": int(person_cat_id),
                "keypoints": kps,
                "score": float(scores[i]) if i < len(scores) else 0.0
            })

    t_pipeline_end = time.perf_counter()

    if total_frames > 0:
        infer_fps = total_frames / total_infer_time if total_infer_time > 0 else 0.0
        pipeline_fps = total_frames / (t_pipeline_end - t_pipeline_start) if (t_pipeline_end - t_pipeline_start) > 0 else 0.0
        avg_infer_ms = (total_infer_time / total_frames) * 1000.0
        print("\n=== Speed summary (dataset pass, excluding COCOeval) ===")
        print(f"Frames processed : {total_frames}")
        print(f"Inference-only   : {infer_fps:.2f} FPS | {avg_infer_ms:.2f} ms / image")
        print(f"Pipeline total   : {pipeline_fps:.2f} FPS | wall {t_pipeline_end - t_pipeline_start:.2f} s")
    else:
        print("No frames processed.")

    # --- COCOeval ---
    pred_json = ROOT / "predictions_ochuman.json"
    pred_json.write_text(json.dumps(results), encoding='utf-8')

    coco_dt = coco_gt.loadRes(str(pred_json))
    e = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    e.evaluate(); e.accumulate(); e.summarize()

if __name__ == "__main__":
    main()
