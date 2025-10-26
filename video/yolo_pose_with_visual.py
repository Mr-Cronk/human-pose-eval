from ultralytics import YOLO
import cv2
import time
import torch
import os

# ---------- pliki i parametry ----------
#VIDEO_FILE = r'C:/Users/I.Kharashun/mmpose/video/solo_train.mp4'
VIDEO_FILE = r'C:/Users/I.Kharashun/mmpose/video/white_2people_2160_3840_24fps_sh.mp4'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[INFO] Używane urządzenie: {DEVICE}')

# ---------- model ----------
model = YOLO('yolo11x-pose.pt').to(DEVICE)

# ---------- wejście wideo ----------
cap = cv2.VideoCapture(VIDEO_FILE)
assert cap.isOpened(), f'Nie udało się otworzyć pliku: {VIDEO_FILE}'

# ---------- kontrolny pomiar jednej klatki ----------
ret, frame = cap.read()
assert ret, 'Nie udało się odczytać pierwszej klatki'
start_single = time.perf_counter()
results = model(frame, verbose=False)
end_single = time.perf_counter()
print(f'[INFO] Czas pojedynczej klatki: {(end_single-start_single)*1000:.2f} ms')

# ---------- pętla wideo + FPS ----------
total_frames, inf_time = 0, 0.0
start_all = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #inference
    t0 = time.perf_counter()
    results = model(frame, verbose=False)
    inf_time += time.perf_counter() - t0
    total_frames += 1

    #wizualizacja
    annotated = results[0].plot()
    cv2.imshow('YOLO-Pose', annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
elapsed = time.perf_counter() - start_all

print(f'\n[INFO] Liczba klatek:              {total_frames}')
print(f'[INFO] FPS (tylko inference):      {total_frames/inf_time:.2f}')
print(f'[INFO] FPS (z over-headem I/O):     {total_frames/elapsed:.2f}')