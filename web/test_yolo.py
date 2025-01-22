import cv2
from ultralytics import YOLO


def process_video(input_video_path, output_video_path, model_path):
    # 載入自訂的 YOLOv8 模型
    model = YOLO(model_path)  # 載入你自己的 YOLOv8 模型

    # 打開影片檔案
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("無法打開影片檔案")
        return

    # 讀取影片的基本屬性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定義影片輸出的編碼格式與儲存檔案
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用自訂 YOLOv8 模型進行物件檢測
        results = model(frame)

        # 取得檢測結果並渲染 (針對第一幀結果)
        annotated_frame = results[0].plot()  # 取得處理過的影像

        # 寫入處理過的畫面
        out.write(annotated_frame)

        # 顯示處理過的畫面 (可選)
        cv2.imshow('Processed Video', annotated_frame)

        # 如果按下 'q' 鍵則退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# 測試程式
input_video = 'D:\\project_Main\\web\\uploads\\Squat(test6).mp4'  # 輸入影片路徑
output_video = 'D:\\project_Main\\web\\output'  # 輸出處理後影片路徑
model_path = 'D:\\project_Main\\modles\\yolov8_squat_model\\weights\\best.pt'  # 你的自訂 YOLO 模型路徑

process_video(input_video, output_video, model_path)
