from pathlib import Path
import json
import time
import numpy as np
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(r"C:/Users/I.Kharashun/yolo/HumanArt")
IMDIR = ROOT / "images" / "val2017"
ANN = ROOT / "annotations" / "validation_humanart_val2017.json"  # COCO GT

MODEL = "yolo11x-pose.pt"  
DEVICE = 0                
IMG_SIZE = 640
DO_WARMUP = True         

def main():

    if not ANN.exists():
        raise FileNotFoundError(f"COCO annotations not found: {ANN}")
    coco_gt = COCO(str(ANN))
    img_ids = coco_gt.getImgIds()

    model = YOLO(MODEL)

    if DO_WARMUP and len(img_ids) > 0:
        warm_img_info = coco_gt.loadImgs(img_ids[0])[0]
        warm_img_path = IMDIR / Path(warm_img_info["file_name"]).name
        if warm_img_path.exists():
            _ = model.predict(source=str(warm_img_path),
                              imgsz=IMG_SIZE, device=DEVICE, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    results = []

    total_frames = 0
    total_infer_time = 0.0

    loop_t0 = time.perf_counter()

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = IMDIR / Path(img_info["file_name"]).name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # czysty czas
        t0 = time.perf_counter()
        pred = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )[0]
        # synchronizacja z GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_infer_time += (t1 - t0)
        # ------------------------------------------

        total_frames += 1

        if pred.keypoints is None or len(pred.keypoints) == 0:
            continue

        kps_xy = pred.keypoints.xy          # (N, K, 2)
        kps_conf = getattr(pred.keypoints, "conf", None)  # (N, K) or None
        scores = pred.boxes.conf.cpu().numpy() if pred.boxes is not None else np.zeros((len(kps_xy),), float)

        for i in range(len(kps_xy)):
            xy = kps_xy[i].cpu().numpy()  # (K, 2), K=17
            if kps_conf is not None:
                vis = (pred.keypoints.conf[i].cpu().numpy() > 0).astype(int) * 2  # >0 ->  (v=2)
            else:
                vis = np.full((xy.shape[0],), 2, dtype=int)

            kps = []
            for (x, y), v in zip(xy, vis):
                kps += [float(x), float(y), int(v)]

            results.append({
                "image_id": int(img_id),
                "category_id": 1,          # person
                "keypoints": kps,          # [x1,y1,v1, ... xK,yK,vK]
                "score": float(scores[i]) if i < len(scores) else 0.0
            })

    loop_t1 = time.perf_counter()

    if total_frames > 0:
        fps_infer = total_frames / total_infer_time if total_infer_time > 0 else 0.0
        fps_pipeline = total_frames / (loop_t1 - loop_t0) if (loop_t1 - loop_t0) > 0 else 0.0
        avg_infer_ms = (total_infer_time / total_frames) * 1000.0 if total_frames else 0.0
        print("\n=== Speed summary (dataset loop, excluding COCOeval) ===")
        print(f"Frames processed: {total_frames}")
        print(f"Inference-only:  {fps_infer:.2f} FPS  | avg {avg_infer_ms:.2f} ms / image")
        print(f"Pipeline total:  {fps_pipeline:.2f} FPS  | wall { (loop_t1 - loop_t0):.2f} s")
    else:
        print("No frames processed.")

    pred_json = ROOT / "predictions_humanart.json"
    pred_json.write_text(json.dumps(results), encoding="utf-8")

    coco_dt = coco_gt.loadRes(str(pred_json))
    e = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    e.evaluate()
    e.accumulate()
    e.summarize()

if __name__ == "__main__":
    main()
