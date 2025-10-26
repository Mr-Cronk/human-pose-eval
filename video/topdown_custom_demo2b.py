import os
import cv2
import time
import numpy as np
import mmcv
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.apis import inference_detector, init_detector
from mmpose.registry import VISUALIZERS
import torch


#torch.set_float32_matmul_precision('medium')

# --- Ścieżki plików ---
DETECTOR_CONFIG = r'C:/Users/I.Kharashun/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
DETECTOR_CHECKPOINT = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

#DETECTOR_CONFIG = r'C:/Users/I.Kharashun/mmpose/demo/mmdetection_cfg/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'
#DETECTOR_CHECKPOINT = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth'

#DETECTOR_CONFIG = r'C:/Users/I.Kharashun/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
#DETECTOR_CHECKPOINT = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'

POSE_CONFIG = r'C:/Users/I.Kharashun/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288.py'
POSE_CHECKPOINT = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288-9a3f7c85_20220914.pth'

#VIDEO_FILE = r'C:/Users/I.Kharashun/mmpose/video/solo_train.mp4'
VIDEO_FILE = r'C:/Users/I.Kharashun/mmpose/video/white_2people_2160_3840_24fps_sh.mp4'
OUTPUT_DIR = r'C:/outvideo'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, os.path.basename(VIDEO_FILE))

# --- Stałe ---
DEVICE = 'cuda:0'
CATEGORY_ID = 0  # 'person' w COCO
BBOX_THR = 0.6
NMS_THR = 0.3
KPT_THR = 0.3
DRAW_HEATMAP = False

# --- Inicjalizacja modeli ---
detector = init_detector(DETECTOR_CONFIG, DETECTOR_CHECKPOINT, device=DEVICE)
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

pose_estimator = init_pose_estimator(
    POSE_CONFIG,
    POSE_CHECKPOINT,
    device=DEVICE,
    cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=DRAW_HEATMAP)))
)

print(next(detector.parameters()).device)
print(next(pose_estimator.parameters()).device)


# --- Wizualizator ---
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.alpha = 0.8
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style='mmpose')

# --- Wczytanie wideo ---
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f'Nie udało się otworzyć pliku: {VIDEO_FILE}')
    exit()


_, frame = cap.read()
det_result = inference_detector(detector.to(DEVICE), frame)
pred = det_result.pred_instances.cpu().numpy()
bboxes = np.concatenate([pred.bboxes, pred.scores[:, None]], axis=1)
bboxes = bboxes[(pred.labels == CATEGORY_ID) & (pred.scores > BBOX_THR)]
bboxes = bboxes[nms(bboxes, NMS_THR), :4]
pose_results = inference_topdown(pose_estimator, frame, bboxes)


start_tot = time.perf_counter()

start_cap = time.perf_counter()
_, frame = cap.read()
end_cap = time.perf_counter()

# Detekcja
start_det = time.perf_counter()
#with torch.inference_mode():
det_result = inference_detector(detector, frame)
end_det = time.perf_counter()

start_box = time.perf_counter()
pred = det_result.pred_instances.cpu().numpy()
bboxes = np.concatenate([pred.bboxes, pred.scores[:, None]], axis=1)
bboxes = bboxes[(pred.labels == CATEGORY_ID) & (pred.scores > BBOX_THR)]
bboxes = bboxes[nms(bboxes, NMS_THR), :4]
end_box = time.perf_counter()


# Pozycje
start_pose = time.perf_counter()
#with torch.inference_mode():
pose_results = inference_topdown(pose_estimator, frame, bboxes)
end_pose = time.perf_counter()

start_merge = time.perf_counter()
data_samples = merge_data_samples(pose_results)
end_merge = time.perf_counter()

end_tot = time.perf_counter()

print(f"[INFO] Cap time: {(end_cap - start_cap)*1000:.2f} ms")
print(f"[INFO] Detector time: {(end_det - start_det)*1000:.2f} ms")
print(f"[INFO] BBox time: {(end_box - start_box)*1000:.2f} ms")
print(f"[INFO] Pose estimator time: {(end_pose - start_pose)*1000:.2f} ms")
print(f"[INFO] Merge time: {(end_merge - start_merge)*1000:.2f} ms")
print(f"[INFO] Total time: {(end_tot - start_tot)*1000:.2f} ms")



# --- Wizualizacja ---
img_rgb = mmcv.bgr2rgb(frame)
visualizer.add_datasample(
    'result',
    img_rgb,
    data_sample=data_samples,
    draw_gt=False,
    draw_heatmap=DRAW_HEATMAP,
    draw_bbox=True,
    show_kpt_idx=False,
    show=False,
    wait_time=0,
    kpt_thr=KPT_THR,
    skeleton_style='mmpose'
)

# --- Pokaż wynik ---
img_out = mmcv.rgb2bgr(visualizer.get_image())
cv2.imshow('Pose Estimation', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()


os.makedirs(OUTPUT_DIR, exist_ok=True)
video_writer = None
total_frames = 0
total_time = 0.0

start_total = time.perf_counter()  # <--- start całkowitego pomiaru

while True:
    ret, frame = cap.read()
    if not ret:
        break
    total_frames += 1
    start = time.perf_counter()

    # --- Detekcja ludzi ---
    det_result = inference_detector(detector.to(DEVICE), frame)
    pred = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate([pred.bboxes, pred.scores[:, None]], axis=1)
    bboxes = bboxes[(pred.labels == CATEGORY_ID) & (pred.scores > BBOX_THR)]
    bboxes = bboxes[nms(bboxes, NMS_THR), :4]

    # --- Estymacja pozy ---
    pose_results = inference_topdown(pose_estimator, frame, bboxes)
    data_samples = merge_data_samples(pose_results)
    total_time += time.perf_counter() - start

end_total = time.perf_counter()  # <--- koniec pomiaru

total_time2=end_total-start_total   
cap.release()


print(f"\n[INFO] Liczba klatek:             {total_frames}")
print(f"[INFO] FPS (czysty inference):    {total_frames / total_time2:.2f}")

