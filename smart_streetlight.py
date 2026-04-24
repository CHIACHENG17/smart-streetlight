from ultralytics import YOLO
import cv2
import time
import numpy as np

# 讀取 YOLO 模型
model = YOLO("yolov8n.pt")

# 讀取影片
video_path = "sample-5s.mp4"
cap = cv2.VideoCapture(video_path)

# 取得影片的幀率和大小
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 車輛類別：COCO資料集
# 2=car, 3=motorcycle, 5=bus, 7=truck
vehicle_classes = [2, 3, 5, 7]

# 時間窗口設定
window_seconds = 30  # 30秒計數窗口
delay_off_seconds = 10  # 偵測到車後延遲10秒才降低亮度

# 狀態變數
vehicle_count_window = 0
start_time = time.time()
last_vehicle_detected_time = None
current_brightness = 30
current_mode = "Low traffic - no vehicle"

# 偵測車輛的緩存（用於去重複計數同一台車）
detected_vehicles_in_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 執行YOLO推理
    results = model(frame, verbose=False)

    current_frame_vehicle_count = 0

    # 畫出偵測到的車輛
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in vehicle_classes:
                current_frame_vehicle_count += 1
                
                # 記錄最後一次偵測到車輛的時間
                last_vehicle_detected_time = time.time()

                # 畫出邊界框
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicle {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 累加當前幀的車輛數
    vehicle_count_window += current_frame_vehicle_count

    # 計算經過的時間
    elapsed_time = time.time() - start_time

    # 檢查是否達到30秒窗口
    if elapsed_time >= window_seconds:
        # 判斷亮度模式
        if vehicle_count_window >= 5:
            current_brightness = 100
            current_mode = "High traffic mode"
            print("✓ 達到5台車！")
        else:
            current_brightness = 30
            current_mode = "Low traffic - no vehicle"

        # 重置計數
        vehicle_count_window = 0
        start_time = time.time()

    # 檢查延遲關燈機制
    if last_vehicle_detected_time is not None:
        time_since_last_vehicle = time.time() - last_vehicle_detected_time
        
        # 如果最近偵測到車輛
        if time_since_last_vehicle < delay_off_seconds:
            # 有車經過，亮度為60%
            if current_brightness != 100:  # 不要覆蓋高流量模式
                current_brightness = 60
                current_mode = "Low traffic - vehicle detected"
        else:
            # 超過延遲時間，回到低亮度
            if current_brightness != 100:  # 不要覆蓋高流量模式
                current_brightness = 30
                current_mode = "Low traffic - no vehicle"

    # ==================== 資訊面板 ====================
    # 建立資訊面板背景
    panel_height = 250
    panel = np.zeros((panel_height, 400, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # 深灰色背景

    # 標題
    cv2.putText(panel, "=== Smart Streetlight ===", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 時間窗口進度
    progress_percentage = (elapsed_time / window_seconds) * 100
    progress_bar_width = int((elapsed_time / window_seconds) * 350)
    cv2.rectangle(panel, (10, 50), (360, 70), (100, 100, 100), -1)
    if progress_bar_width > 0:
        cv2.rectangle(panel, (10, 50), (10 + progress_bar_width, 70), (0, 200, 255), -1)
    cv2.putText(panel, f"Time window: {elapsed_time:.1f}s / {window_seconds}s", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 車輛計數
    cv2.putText(panel, f"Window vehicle count: {vehicle_count_window}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(panel, f"Current frame vehicles: {current_frame_vehicle_count}", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # 亮度顯示
    brightness_color = (0, 0, 255) if current_brightness == 100 else (0, 255, 0) if current_brightness == 60 else (100, 100, 255)
    cv2.rectangle(panel, (10, 165), (390, 195), brightness_color, -1)
    cv2.putText(panel, f"Brightness: {current_brightness}%", (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # 模式顯示
    cv2.putText(panel, f"Mode: {current_mode}", (10, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 延遲倒計時（如果有車經過）
    if last_vehicle_detected_time is not None:
        time_since_vehicle = time.time() - last_vehicle_detected_time
        remaining_time = max(0, delay_off_seconds - time_since_vehicle)
        cv2.putText(panel, f"Delay off: {remaining_time:.1f}s", (200, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

    # 將面板放到影片畫面的右上角
    frame_with_panel = frame.copy()
    frame_with_panel[0:panel_height, 0:400] = panel

    # ==================== 主要視窗顯示 ====================
    cv2.imshow("YOLO Smart Streetlight Simulation", frame_with_panel)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 除錯輸出
    if elapsed_time >= window_seconds - 0.1:  # 在窗口結束時輸出
        print("=" * 50)
        print(f"⏱️  Time window: {window_seconds} seconds")
        print(f"🚗 Vehicle count in window: {vehicle_count_window}")
        print(f"💡 Brightness: {current_brightness}%")
        print(f"📍 Mode: {current_mode}")
        print("=" * 50)

cap.release()
cv2.destroyAllWindows()
