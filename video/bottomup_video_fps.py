import os
import time
import cv2
import mmcv
import mmengine
import numpy as np
import torch

from mmpose.apis import init_model as init_pose_estimator, inference_bottomup
from mmpose.structures import merge_data_samples
from mmpose.registry import VISUALIZERS

POSE_CONFIG     = r"C:/Users/I.Kharashun/mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-s_8xb32-600e_coco-640x640.py"
POSE_CHECKPOINT = r"https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth"

# Wejście
USE_WEBCAM   = False       
WEBCAM_INDEX = 0
INPUT_VIDEO = r'C:/Users/I.Kharashun/mmpose/video/white_2people_2160_3840_24fps_sh.mp4'


OUTPUT_ROOT  = r"C:/outvideo"
SAVE_VIDEO   = True          
OUTPUT_FPS   = 25
FOURCC       = "mp4v"

# Urządzenie
DEVICE       = "cuda:0"       # np. "cuda:0" lub "cpu"

ENABLE_VIS   = True          
SHOW_WINDOW  = False         
DRAW_HEATMAP = False
KPT_THR      = 0.3
RADIUS       = 3
THICKNESS    = 1

SAVE_PREDICTIONS = False
PRED_JSON_NAME   = "bottomup_predictions.json"



def main():
    pose_estimator = init_pose_estimator(
        POSE_CONFIG,
        POSE_CHECKPOINT,
        device=DEVICE,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=DRAW_HEATMAP))),
    )

    visualizer = None
    if ENABLE_VIS:
        pose_estimator.cfg.visualizer.radius = RADIUS
        pose_estimator.cfg.visualizer.line_width = THICKNESS
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style="mmpose")

    # Wejście
    if USE_WEBCAM:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        out_name = "webcam.mp4"
    else:
        cap = cv2.VideoCapture(INPUT_VIDEO)
        out_name = os.path.basename(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {'webcam' if USE_WEBCAM else INPUT_VIDEO}")

    # Wyjście (wideo)
    writer = None
    out_path = None
    if ENABLE_VIS and SAVE_VIDEO and OUTPUT_ROOT:
        mmengine.mkdir_or_exist(OUTPUT_ROOT)
        out_path = os.path.join(OUTPUT_ROOT, out_name if USE_WEBCAM else out_name)
        if out_path.lower().endswith(".mp4") is False:
            out_path += ".mp4"

    # Liczniki FPS (tylko inferencja)
    total_frames = 0
    total_inf_time = 0.0

    pred_instances_list = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            # --- pomiar czystej inferencji bottom-up ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pose_results = inference_bottomup(pose_estimator, frame)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_inf_time += (t1 - t0)
            # ------------------------------------------

            data_samples = merge_data_samples(pose_results)

            if SAVE_PREDICTIONS:
                pred_instances_list.append(
                    dict(frame_id=total_frames, instances=data_samples.get("pred_instances", None))
                )

            # Wizualizacja (nie wliczana do FPS)
            if ENABLE_VIS:
                img_rgb = mmcv.bgr2rgb(frame)
                visualizer.add_datasample(
                    "result",
                    img_rgb,
                    data_sample=data_samples,
                    draw_gt=False,
                    draw_heatmap=DRAW_HEATMAP,
                    draw_bbox=False,  # bottom-up bez bboxes osób
                    show=SHOW_WINDOW,
                    wait_time=1e-3 if SHOW_WINDOW else 0,
                    kpt_thr=KPT_THR,
                )
                vis = mmcv.rgb2bgr(visualizer.get_image())

                if writer is None and out_path:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
                    writer = cv2.VideoWriter(out_path, fourcc, OUTPUT_FPS, (w, h))
                if writer is not None:
                    writer.write(vis)

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    fps = (total_frames / total_inf_time) if total_inf_time > 0 else 0.0
    print(f"[INFO] Frames processed        : {total_frames}")
    print(f"[INFO] Inference time (total)  : {total_inf_time:.2f} s")
    print(f"[INFO] FPS (inference-only)    : {fps:.2f}")

    if SAVE_PREDICTIONS and OUTPUT_ROOT:
        try:
            import json_tricks as json
            pred_path = os.path.join(OUTPUT_ROOT, PRED_JSON_NAME)
            with open(pred_path, "w") as f:
                json.dump(
                    dict(meta_info=pose_estimator.dataset_meta, instance_info=pred_instances_list),
                    f,
                    indent=2,
                )
            print(f"[INFO] Predictions saved to: {pred_path}")
        except Exception as e:
            print(f"[WARN] Could not save predictions: {e}")

    if out_path:
        print(f"[INFO] Output video saved to   : {out_path}")


if __name__ == "__main__":
    main()
