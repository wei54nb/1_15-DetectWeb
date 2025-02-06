from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Response, jsonify
from flask_socketio import SocketIO
import eventlet
import eventlet.wsgi
import os
import cv2
import numpy as np
import logging
from ultralytics import YOLO
import mediapipe as mp
from werkzeug.utils import secure_filename
import torch
import queue
import threading
import time

eventlet.monkey_patch()  # 這行確保 socketIO 可以正確使用 eventlet
app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# 全局變數用於控制偵測狀態
detection_active = False
current_exercise_type = 'squat'
frame_buffer = queue.Queue(maxsize=2)
processed_frame_buffer = queue.Queue(maxsize=2)

#計算次數
exercise_count = 0
last_pose = None
mid_pose_detected = False


# 日誌設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe 設定
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# 文件儲存
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# 模型位置
MODEL_PATHS = {
    'squat': 'D:\\project_Main\\modles\\yolov8_squat_model\\weights\\best.pt',
    'bicep-curl': 'D:\\project_Main\\modles\\best.pt',
    'shoulder-press': 'D:\\project_Main\\modles\\yolov8_shoulder_model\\weights\\best.pt',
    'push-up': 'D:\\project_Main\\modles\\push-up_model\\weights\\pushup_best.pt'
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 加載模型
models = {}
for exercise_type, model_path in MODEL_PATHS.items():
    try:
        model = YOLO(model_path).to('cuda')
        models[exercise_type] = model
        logger.info(f" YOLO model for {exercise_type} loaded successfully on GPU")

        # 紀錄模型的類別
        logger.info(f" {exercise_type} 模型類別: {model.names}")

    except Exception as e:
        logger.error(f" Error loading YOLO model for {exercise_type}: {e}")

# 工具函數
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle


def get_exercise_angles(landmarks):
    angles = {}
    try:
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angles['左手肘'] = calculate_angle(left_shoulder, left_elbow, left_wrist)

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angles['右手肘'] = calculate_angle(right_shoulder, right_elbow, right_wrist)

        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        angles['左膝蓋'] = calculate_angle(left_hip, left_knee, left_ankle)

        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        angles['右膝蓋'] = calculate_angle(right_hip, right_knee, right_ankle)

        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        angles['左肩膀'] = calculate_angle(left_hip, left_shoulder, left_elbow)

        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        angles['右肩膀'] = calculate_angle(right_hip, right_shoulder, right_elbow)

        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        angles['左髖部'] = calculate_angle(left_shoulder, left_hip, left_knee)

        # 右髖部角度
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        angles['右髖部'] = calculate_angle(right_shoulder, right_hip, right_knee)

    except Exception as e:
        logger.error(f"Error calculating angles: {e}")

    return angles


def process_frame_realtime(frame, exercise_type):
    global exercise_count, last_pose, mid_pose_detected

    try:
        frame = cv2.resize(frame, (360, 360))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = frame.copy()
        model = models.get(exercise_type)

        if model:
            results_yolo = model(frame, conf=0.3)

            if len(results_yolo[0].boxes) > 0:
                best_box = results_yolo[0].boxes[0]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                conf = float(best_box.conf)
                class_id = int(best_box.cls)
                class_name = model.names[class_id]

                # 記錄 YOLO 偵測的動作
                logger.info(f"[YOLO] Detected {class_name} with confidence {conf:.2f} in {exercise_type} mode")

                # 繪製框框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # MediaPipe 姿勢偵測
                with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=0) as pose:
                    results_pose = pose.process(frame_rgb)

                    if results_pose.pose_landmarks:
                        landmarks = results_pose.pose_landmarks.landmark
                        angles = get_exercise_angles(landmarks)

                        # 傳送角度數據到前端
                        socketio.emit('angle_data', angles)
                        logger.info(f"[Angle] Sent angles: {angles}")

                # **計數邏輯**
                num_classes = len(model.names)  # 取得該模型的類別數量

                if num_classes == 1:
                    # 只有一個類別的情況（如 bicep-curl、shoulder-press）
                    if class_id == 0:  # 確保是該動作
                        exercise_count += 1
                        logger.info(f"[Counter] {exercise_type} count updated: {exercise_count}")
                        socketio.emit('exercise_count_update', {'count': exercise_count})

                elif num_classes == 2:
                    # 兩個類別的情況（如 squat, push-up）
                    if last_pose is not None:
                        if last_pose == 0 and class_id == 1:
                            mid_pose_detected = True
                        elif last_pose == 1 and class_id == 0 and mid_pose_detected:
                            exercise_count += 1
                            mid_pose_detected = False
                            logger.info(f"[Counter] {exercise_type} count updated: {exercise_count}")
                            socketio.emit('exercise_count_update', {'count': exercise_count})

                    last_pose = class_id  # 更新上一個姿勢


                # 在畫面上顯示運動次數
                cv2.putText(annotated_frame,
                            f'Count: {exercise_count}',
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

        return annotated_frame

    except Exception as e:
        logger.error(f"處理幀時發生錯誤: {e}")
        return frame



def video_capture_thread(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("無法開啟攝像頭")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 20)

    while detection_active:
        ret, frame = cap.read()
        if not ret:
            logger.error("無法讀取影像幀")
            break
        frame = cv2.resize(frame, (360, 360))

        if not frame_buffer.full():
            frame_buffer.put(frame)
        else:
            # 清空隊列避免延遲
            try:
                frame_buffer.get_nowait()
            except queue.Empty:
                pass
            frame_buffer.put(frame)
        time.sleep(0.01)

    cap.release()
    logger.info("攝像頭執行緒已正常停止")

def frame_processing_thread(exercise_type='squat'):
    while detection_active:
        if not frame_buffer.empty():
            frame = frame_buffer.get()
            processed_frame = process_frame_realtime(frame, exercise_type)
            if not processed_frame_buffer.full():
                processed_frame_buffer.put(processed_frame)
        time.sleep(0.001)
    logger.info("畫面處理執行緒已停止")

def generate_frames():
    while True:
        if not processed_frame_buffer.empty():
            frame = processed_frame_buffer.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def cleanup_buffers():
    """定期清理緩衝區"""
    while True:
        if frame_buffer.qsize() > 1:
            try:
                frame_buffer.get_nowait()
            except queue.Empty:
                pass
        time.sleep(0.1)

def check_thread_status():
    """檢查執行序狀態"""
    while True:
        active_threads = threading.enumerate()
        logger.info(f"Active threads: {[t.name for t in active_threads]}")
        time.sleep(5)  # 每5秒檢查一次

# 路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')


@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_active, current_exercise_type, exercise_count, last_pose, mid_pose_detected

    try:
        exercise_type = request.args.get('exercise_type', 'squat')  # 確保獲取前端的運動類型
        if exercise_type not in models:
            return jsonify({'success': False, 'error': '不支援的運動類型'}), 400

        current_exercise_type = exercise_type
        exercise_count = 0  # 重置計數器
        last_pose = None
        mid_pose_detected = False

        if not detection_active:
            detection_active = True
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                return jsonify({'success': False, 'error': '無法開啟攝像頭'}), 400
            cap.release()

            # 啟動執行緒
            threads = [
                threading.Thread(target=video_capture_thread, name="VideoCapture"),
                threading.Thread(target=frame_processing_thread, args=(exercise_type,), name="FrameProcessing"),
                threading.Thread(target=cleanup_buffers, daemon=True, name="BufferCleanup")
            ]

            for thread in threads:
                thread.start()
                logger.info(f"Started thread: {thread.name}")

            logger.info(f"✅ 啟動 {exercise_type} 模型進行即時偵測")
            return jsonify({'success': True})

    except Exception as e:
        logger.error(f"啟動偵測時發生錯誤: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False
    logger.info("停止即時偵測執行緒")
    return jsonify({'success': True})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not processed_frame_buffer.empty():
                frame = processed_frame_buffer.get()
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)

    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/static/models/<path:filename>')
def serve_model(filename):
    return send_from_directory('static/models', filename)


if __name__ == '__main__':



    # 確保必要的目錄存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs('static/models', exist_ok=True)

    # 清空緩衝區
    while not frame_buffer.empty():
        frame_buffer.get()
    while not processed_frame_buffer.empty():
        processed_frame_buffer.get()

    threading.Thread(target=check_thread_status,
                    daemon=True,
                    name="ThreadMonitor").start()

    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5000)), app)