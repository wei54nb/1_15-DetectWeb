import eventlet
eventlet.monkey_patch()  # 這行確保 socketIO 可以正確使用 eventlet

# 使用 PyMySQL 代替 MySQLdb（純 Python，不需要編譯 C 擴展）
import pymysql
# 在程式開頭更新日誌配置
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 使用 stdout 而不是 stderr
        logging.FileHandler('pose_detection.log', encoding='utf-8')  # 指定編碼
    ]
)
logger = logging.getLogger(__name__)


from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask import Flask, request, jsonify, render_template
from werkzeug.security import generate_password_hash
import mysql.connector
from mysql.connector import Error
from io import BytesIO
import pandas as pd
from flask import send_file
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Response, jsonify
from flask_socketio import SocketIO
import eventlet.wsgi
import os
import cv2
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import torch
import queue
import threading
import time
import mediapipe as mp
from datetime import datetime
import yaml
from flask_login import UserMixin

# ------------------------------
# 資料庫設定與連線 (使用 PyMySQL)
# ------------------------------
db_config = {
    'host': 'localhost',
    'user': 'nkust_user',
    'password': '1234',
    'database': 'nkust_exercise'
}


# 建立 Flask 應用前先設定資料庫配置到 app.config 中
app = Flask(__name__, static_folder='static')


app.secret_key = "your_secret_key"  # 設定 Flask 的 session key
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, user_id, username, role):
        self.id = str(user_id)  # 🔹 確保 user_id 是字串，避免 session 讀取問題
        self.username = username
        self.role = role


# 自訂一個函式用來取得資料庫連線
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
            app.logger.error("資料庫使用者名稱或密碼錯誤")
        elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
            app.logger.error("資料庫不存在")
        else:
            app.logger.error(f"資料庫連接錯誤: {err}")
        return None


def user_exists(username):
    """檢查用戶名是否已存在"""
    conn = get_db_connection()
    if not conn:
        raise Exception("無法連接資料庫")

    try:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        count = cursor.fetchone()[0]
        return count > 0
    finally:
        cursor.close()
        conn.close()


def create_user(username, password_hash, role):
    """創建新用戶"""
    conn = get_db_connection()
    if not conn:
        raise Exception("無法連接資料庫")

    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO users (username, password_hash, role)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (username, password_hash, role))
        conn.commit()
    except Error as e:
        conn.rollback()
        raise Exception(f"創建用戶錯誤: {e}")
    finally:
        cursor.close()
        conn.close()

def test_db_connection():
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return result[0] == 1
        return False
    except Exception as e:
        print(f"測試連接錯誤: {e}")
        return False

def check_users_table():
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("DESCRIBE users")
            columns = cursor.fetchall()
            cursor.close()
            conn.close()
            return len(columns) > 0
        return False
    except Exception as e:
        print(f"檢查表結構錯誤: {e}")
        return False

# 在啟動應用前測試
if not test_db_connection():
    print("警告: 無法連接到資料庫!")


socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# ------------------------------
# 其他設定與模型初始化
# ------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定 GPU

# 檢查 GPU
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("Using CPU")

# 在全局變量區域初始化姿態估計模型
pose_model = YOLO("yolov8n-pose.pt")  # 使用基礎姿態估計模型

# 全局變數用於控制偵測狀態
detection_active = False
current_exercise_type = 'squat'
frame_buffer = queue.Queue(maxsize=2)
processed_frame_buffer = queue.Queue(maxsize=2)
exercise_count = 0
last_pose = None
mid_pose_detected = False

# 日誌設定（若重複設定則可略過）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 文件儲存設定
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 模型位置設定
MODEL_PATHS = {
    'squat': 'D:\\project_Main\\modles\\yolov8_squat_model\\weights\\best.pt',
    'bicep-curl': 'D:\\project_Main\\modles\\best.pt',
    'shoulder-press': 'D:\\project_Main\\modles\\yolov8_shoulder_model\\weights\\best.pt',
    'push-up': 'D:\\project_Main\\modles\\push-up_model\\weights\\pushup_best.pt',
    'pull-up': 'D:\\project_Main\\modles\\best_pullup.pt',
    'dumbbell-row':'D:\\project_Main\\modles\\dumbbellrow_train\\weights\\best.pt'

}

# 加載運動分類模型
with app.app_context():
    models = {}
    for exercise_type, model_path in MODEL_PATHS.items():
        try:
            if torch.cuda.is_available():
                exercise_model = YOLO(model_path).to('cuda')
            else:
                exercise_model = YOLO(model_path)
            models[exercise_type] = exercise_model
            logger.info(f"YOLO model for {exercise_type} loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading YOLO model for {exercise_type}: {e}")


# 工具函數
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_angle(a, b, c):
    # 將 a, b, c 轉換為 numpy 陣列
    a, b, c = np.array(a), np.array(b), np.array(c)

    # 計算向量 BA 和 BC（即從 b 到 a 以及從 b 到 c 的向量）
    ba = a - b
    bc = c - b

    # 計算向量的點積
    dot_product = np.dot(ba, bc)

    # 計算向量的長度
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # 防止除以 0 的情況（如果某向量長度為 0，就直接返回 0 度）
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    # 計算夾角的 cosine 值，並利用 clip 限制範圍在 [-1, 1]
    cos_theta = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)

    # 利用 arccos 求出角度，再轉換為度數
    angle = np.degrees(np.arccos(cos_theta))

    return angle


def get_pose_angles(keypoints):
    angles = {}
    try:
        logger.info("get_pose_angles function called")  # 添加這行
        # Check if enough keypoints are detected
        if len(keypoints) < 17:
            logger.warning("Not enough keypoints detected to calculate angles.")
            return angles  # Return empty dictionary if not enough keypoints

        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        left_elbow = keypoints[7][:2]
        right_elbow = keypoints[8][:2]
        left_wrist = keypoints[9][:2]
        right_wrist = keypoints[10][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]
        left_ankle = keypoints[15][:2]
        right_ankle = keypoints[16][:2]

        # 檢查關鍵點座標是否有效
        if any(np.isnan(kp).any() for kp in [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            logger.warning("Invalid keypoint coordinates detected.")
            return angles

        # 計算基本角度
        try:
            angles['左手肘'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
            angles['右手肘'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angles['左膝蓋'] = calculate_angle(left_hip, left_knee, left_ankle)
            angles['右膝蓋'] = calculate_angle(right_hip, right_knee, right_ankle)
            angles['左肩膀'] = calculate_angle(left_hip, left_shoulder, left_elbow)
            angles['右肩膀'] = calculate_angle(right_hip, right_shoulder, right_elbow)
            angles['左髖部'] = calculate_angle(left_shoulder, left_hip, left_knee)
            angles['右髖部'] = calculate_angle(right_shoulder, right_hip, right_knee)
        except Exception as e:
            logger.error(f"Error calculating angles: {e}")

        logger.info(f"Calculated angles: {angles}")  # 添加這行
    except Exception as e:
        logging.error(f"計算角度時發生錯誤: {e}")
    logger.info(f"Angles calculated: {angles}")
    return angles

def process_frame(frame):
    frame = cv2.resize(frame, (480, 480))
    results = pose_model(frame)  # 改為使用 pose_model
    annotated_frame = frame.copy()

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()[0]  # 獲取關鍵點座標
            angles = get_pose_angles(keypoints)
            socketio.emit('angle_data', angles)
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
    return annotated_frame

def get_exercise_angles(landmarks, exercise_type='squat'):
    angles = {}
    try:
        # 基本姿勢點
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # 計算基本角度
        angles['左手肘'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
        angles['右手肘'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
        angles['左膝蓋'] = calculate_angle(left_hip, left_knee, left_ankle)
        angles['右膝蓋'] = calculate_angle(right_hip, right_knee, right_ankle)
        angles['左肩膀'] = calculate_angle(left_hip, left_shoulder, left_elbow)
        angles['右肩膀'] = calculate_angle(right_hip, right_shoulder, right_elbow)
        angles['左髖部'] = calculate_angle(left_shoulder, left_hip, left_knee)
        angles['右髖部'] = calculate_angle(right_shoulder, right_hip, right_knee)

        # 引體向上專用角度計算
        if exercise_type == 'pull-up':
            # 計算手臂與垂直線的夾角
            # 創建垂直參考點 (與肩同x,但y較小)
            left_vertical = [left_shoulder[0], left_shoulder[1] - 0.2]
            right_vertical = [right_shoulder[0], right_shoulder[1] - 0.2]

            angles['左手臂懸垂角度'] = calculate_angle(left_vertical, left_shoulder, left_elbow)
            angles['右手臂懸垂角度'] = calculate_angle(right_vertical, right_shoulder, right_elbow)

            # 計算身體傾斜角度
            # 創建垂直參考點 (與髖部同x,但y較小)
            hip_vertical = [left_hip[0], left_hip[1] - 0.2]
            angles['身體傾斜度'] = calculate_angle(hip_vertical, left_hip, left_shoulder)

            # 計算肘部彎曲程度
            angles['引體向上深度'] = min(angles['左手肘'], angles['右手肘'])

    except Exception as e:
        logger.error(f"Error calculating angles: {e}")

    return angles


exercise_count = 0
squat_state = "up"  # 深蹲初始狀態：站立（"up"）
last_squat_time = 0  # 記錄上一次成功計數的時間（秒）
last_pose = None
mid_pose_detected = False
def process_frame_realtime(frame, exercise_type):
    global exercise_count, last_pose, mid_pose_detected, squat_state, last_squat_time

    try:
        # 步驟1: 基礎影像處理（調整解析度）
        frame = cv2.resize(frame, (480, 480))
        annotated_frame = frame.copy()

        # 步驟2: 使用 YOLOv8-pose 進行姿態估計
        # 請確保這裡使用的是分離出的姿態模型 pose_model
        pose_results = pose_model(frame, conf=0.3, verbose=True)
        if not pose_results or len(pose_results) == 0:
            logger.warning("❌ YOLO pose detection returned empty results!")
            return frame

        # 處理姿態估計結果並計算角度
        angles = {}
        if pose_results and len(pose_results) > 0 and pose_results[0].keypoints is not None:
            # 獲取關鍵點座標
            keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]
            logger.info(f"取得關鍵點數量: {len(keypoints)}")

            if len(keypoints) >= 17:
                # 提取主要關鍵點（座標：x, y）
                left_shoulder  = keypoints[5][:2]
                right_shoulder = keypoints[6][:2]
                left_elbow     = keypoints[7][:2]
                right_elbow    = keypoints[8][:2]
                left_wrist     = keypoints[9][:2]
                right_wrist    = keypoints[10][:2]
                left_hip       = keypoints[11][:2]
                right_hip      = keypoints[12][:2]
                left_knee      = keypoints[13][:2]
                right_knee     = keypoints[14][:2]
                left_ankle     = keypoints[15][:2]
                right_ankle    = keypoints[16][:2]

                # 計算各關節角度
                angles['左手肘'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
                angles['右手肘'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
                angles['左膝蓋'] = calculate_angle(left_hip, left_knee, left_ankle)
                angles['右膝蓋'] = calculate_angle(right_hip, right_knee, right_ankle)
                angles['左肩膀'] = calculate_angle(left_hip, left_shoulder, left_elbow)
                angles['右肩膀'] = calculate_angle(right_hip, right_shoulder, right_elbow)
                angles['左髖部'] = calculate_angle(left_shoulder, left_hip, left_knee)
                angles['右髖部'] = calculate_angle(right_shoulder, right_hip, right_knee)

                # 將角度數據發送到前端
                socketio.emit('angle_data', angles)
                logger.info(f"Angles calculated and emitted: {angles}")

                # ───────────────────────────────
                # 深蹲計數：採用膝關節角度作為判斷依據（加入冷卻期）
                if exercise_type == "squat":
                    # 以左右膝蓋角度平均值作為判斷依據
                    avg_knee_angle = (angles['左膝蓋'] + angles['右膝蓋']) / 2.0
                    logger.info(f"Average knee angle: {avg_knee_angle:.1f}")

                    # 調整閾值：當從站立(up)變為下蹲(down)時，膝角低於 85°
                    if squat_state == "up" and avg_knee_angle < 85:
                        squat_state = "down"
                        logger.info("Down position detected")
                    # 當從下蹲狀態(down)回到站立(up)時，膝角大於 155°，並且冷卻期超過 0.3 秒
                    elif squat_state == "down" and avg_knee_angle > 155:
                        current_time = time.time()
                        if current_time - last_squat_time > 0.3:
                            squat_state = "up"
                            exercise_count += 1
                            last_squat_time = current_time
                            logger.info(f"Squat completed, count incremented to {exercise_count}")
                            socketio.emit('exercise_count_update', {'count': exercise_count})
                # ───────────────────────────────


        else:
            logger.warning("❌ YOLO pose detection returned empty results!")

        # 若其他運動需要使用分類模型計數，可保留以下邏輯，但建議與角度判斷分離以避免重複計數
        current_model = models.get(exercise_type)
        if current_model:
            exercise_results = current_model(frame, conf=0.5)
            logger.info(f"運動分類結果：檢測到 {len(exercise_results[0].boxes)} 個框")
            if len(exercise_results[0].boxes) > 0:
                best_box = exercise_results[0].boxes[0]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
                conf = float(best_box.conf)
                class_id = int(best_box.cls)
                class_name = current_model.names[class_id]
                # 繪製檢測框與標籤
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # 若不是深蹲模式，執行分類計數邏輯（深蹲計數依然採用角度判斷）
                if exercise_type != "squat":
                    num_classes = len(current_model.names)
                    if num_classes == 1:
                        if class_id == 0:
                            exercise_count += 1
                            socketio.emit('exercise_count_update', {'count': exercise_count})
                    elif num_classes == 2:
                        if last_pose is not None:
                            if last_pose == 0 and class_id == 1:
                                mid_pose_detected = True
                            elif last_pose == 1 and class_id == 0 and mid_pose_detected:
                                exercise_count += 1
                                mid_pose_detected = False
                                socketio.emit('exercise_count_update', {'count': exercise_count})
                        last_pose = class_id

        # 顯示最終運動計數
        cv2.putText(annotated_frame,
                    f'Count: {exercise_count}',
                    (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        return annotated_frame

    except Exception as e:
        logger.error(f"Error in process_frame_realtime: {e}")
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
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

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

def process_video(input_video_path, output_video_path, exercise_type):

    detection_info = []  # Initialize outside the try block
    frame_count = 0
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file {input_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use avc1 codec for H.264
    out = None
    try:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise Exception(f"Error creating video writer for {output_video_path}")
        detection_info = []

        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
        ) as pose:

            model = models.get(exercise_type)
            if model is None:
                raise Exception(f"Model for {exercise_type} not found")

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results_yolo = model(frame, conf=0.25)
                    annotated_frame = frame.copy()  # 使用原始帧的副本
                    results_pose = pose.process(frame_rgb)

                    frame_info = {
                        'frame': frame_count,
                        'angles': {}
                    }

                    if results_pose.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            results_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )

                        landmarks = results_pose.pose_landmarks.landmark
                        angles = get_exercise_angles(landmarks, exercise_type)
                        if not angles and exercise_type == 'pull-up':
                            logger.warning(f"Frame {frame_count}: No angles calculated for pull-up exercise")

                        frame_info['angles'] = angles


                    if len(results_yolo[0].boxes) > 0:
                        for box in results_yolo[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf)
                            class_id = int(box.cls)
                            class_name = model.names[class_id]

                            # 繪製邊界框
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                                          2)
                            # 添加標籤：類別名稱和置信度
                            label = f'{class_name} {conf:.2f}'
                            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detection_info.append(frame_info)
                    out.write(annotated_frame)
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    if out:
                        out.write(frame)
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"Processing frame {frame_count}/{total_frames}")
        if out:
            out.release()
        cap.release()
        torch.cuda.empty_cache()
        logger.info(f"Video processing completed: {output_video_path}")
        return detection_info, fps

    except Exception as e:
        logger.error(f"Error in process_video: {e}")
        raise

def setup_gpu():
    try:
        if torch.cuda.is_available():
            print(f"PyTorch 可以使用 GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()  # 清空 GPU 緩存
        else:
            print("PyTorch 無法使用 GPU，將使用 CPU")
    except Exception as e:
        print(f"GPU 配置時發生錯誤：{e}")
        print("將使用 CPU 運行")


@login_manager.user_loader
def load_user(user_id):
    print(f"🔍 嘗試加載用戶 ID: {user_id}")  # Debug

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()

    print(f"📝 查詢結果: {user}")  # Debug

    if user:
        return User(user["user_id"], user["username"], user["role"])
    else:
        print("⚠️ 找不到用戶，回傳 None")
        return None  # 確保找不到用戶時回傳 None


@app.route('/')
def index():
    return render_template('index.html', current_user=current_user)

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/Equipment_Introduction')
def Equipment_Introduction():
    return render_template('Equipment Introduction Page.html')

@app.route('/Exercise_Knowledge')
def Exercise_Knowledge():
    return render_template('Exercise Knowledge Page.html')

@app.route('/page1')
def Recommended_Setup_Page():
    return render_template('Recommended_Setup_Page.html')

@app.route('/page2')
def Technologies_Page():
    return render_template('Technologies_Page.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_active, current_exercise_type, exercise_count, last_pose, mid_pose_detected, remaining_sets

    try:
        exercise_type = request.args.get('exercise_type', 'squat')

        data = request.get_json() or {}
        weight = data.get('weight')
        reps = data.get('reps')  # 每組次數
        sets = data.get('sets')  # 組數
        student_id = data.get('student_id')

        if not all([student_id, weight, reps, sets]):
            return jsonify({'success': False, 'error': '請完整填寫所有輸入欄位'}), 400

        connection = get_db_connection()
        cursor = connection.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("""
            INSERT INTO exercise_info (student_id, weight, reps, sets, exercise_type, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (student_id, weight, reps, sets, exercise_type, timestamp))
        connection.commit()
        cursor.close()
        connection.close()

        current_exercise_type = exercise_type
        exercise_count = 0
        last_pose = None
        mid_pose_detected = False

        remaining_sets = int(sets)  # 記錄剩餘組數
        socketio.emit('remaining_sets_update', {'remaining_sets': remaining_sets})  # 傳送剩餘組數到前端

        if not detection_active:
            detection_active = True
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                return jsonify({'success': False, 'error': '無法開啟攝像頭'}), 400
            cap.release()
            threading.Thread(target=video_capture_thread, name="VideoCapture").start()
            threading.Thread(target=frame_processing_thread, args=(exercise_type,), name="FrameProcessing").start()
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

@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return redirect(url_for('index'))
        file = request.files['file']
        exercise_type = request.form.get('exercise', 'squat')
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                output_filename = f"processed_{filename}"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                file.save(input_path)
                logger.info(f"File saved: {input_path}")
                detection_info, fps = process_video(input_path, output_path, exercise_type)
                logger.info(f"Video processed: {output_path}")
                if not os.path.exists(output_path):
                    logger.error(f"Output file not found: {output_path}")
                    return "Error: Processed video not found", 500
                torch.cuda.empty_cache()
                return render_template('uploaded.html',
                                       filename=output_filename,
                                       detection_info=detection_info,
                                       fps=fps)
            except Exception as e:
                torch.cuda.empty_cache()
                logger.error(f"Error processing video: {e}")
                return f"Error processing video: {str(e)}", 500
        else:
            logger.warning("Invalid file type")
            return "Invalid file type", 400
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        logger.info(f"Attempting to stream file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return "File not found", 404
        return send_from_directory(
            app.config['OUTPUT_FOLDER'],
            filename,
            mimetype='video/mp4'
        )
    except Exception as e:
        logger.error(f"Error streaming video: {e}")
        return f"Error streaming video: {str(e)}", 500


@app.route('/export_excel', methods=['GET'])
def export_excel():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT student_id, weight, reps, sets, exercise_type, timestamp
            FROM exercise_info
            ORDER BY timestamp DESC
        """)

        records = cursor.fetchall()
        if not records:
            return jsonify({'error': '沒有找到運動記錄'}), 404

        df = pd.DataFrame(records)
        df.columns = ['學號', '重量(Kg)', '每組次數', '組數', '運動類型', '紀錄時間']
        df['紀錄時間'] = pd.to_datetime(df['紀錄時間']).dt.strftime('%Y-%m-%d %H:%M:%S')

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='運動紀錄')

            workbook = writer.book
            worksheet = writer.sheets['運動紀錄']
            worksheet.set_column('A:A', 15)
            worksheet.set_column('B:B', 10)
            worksheet.set_column('C:C', 15)
            worksheet.set_column('D:D', 10)
            worksheet.set_column('E:E', 15)
            worksheet.set_column('F:F', 20)

            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#00BCD4',
                'font_color': 'white',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            worksheet.freeze_panes(1, 0)

        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True, download_name=f"運動紀錄_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    except Exception as e:
        return jsonify({'error': f'匯出記錄失敗: {str(e)}'}), 500

@app.route('/update_monster_count', methods=['POST'])
def update_monster_count():
    try:
        data = request.json
        count = data.get('count')
        exercise_type = data.get('exercise_type')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO exercise_records (monster_count, exercise_type, timestamp)
            VALUES (%s, %s, %s)
        """, (count, exercise_type, timestamp))
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"更新紀錄發生錯誤: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_db')
def test_db():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute('SELECT 1')
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        return jsonify({'message': '數據庫連接成功！', 'result': result})
    except Exception as e:
        return jsonify({'error': f'數據庫連接失敗: {str(e)}'})


@app.route('/classroom')
@login_required  # 確保用戶已登入
def classroom():
    return render_template('classroom.html')


@app.route('/api/discussions', methods=['GET'])
def get_discussions():
    try:
        course_id = request.args.get('course_id')
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)  # 使用 dictionary cursor 讓結果更容易處理

        cursor.execute("""
            SELECT d.*, c.course_name,
                   COALESCE(u_student.username, '') as student_username,
                   COALESCE(u_teacher.username, '') as teacher_username,
                   (SELECT COUNT(*) FROM responses WHERE discussion_id = d.discussion_id) as response_count
            FROM discussions d
            JOIN courses c ON d.course_id = c.course_id
            LEFT JOIN users u_student ON d.student_id = u_student.user_id
            LEFT JOIN users u_teacher ON d.teacher_id = u_teacher.user_id
            WHERE d.course_id = %s
            ORDER BY d.created_at DESC
        """, (course_id,))

        discussions = cursor.fetchall()
        cursor.close()
        connection.close()

        # 確保所有數據欄位格式正確
        for d in discussions:
            d['created_at'] = d['created_at'].isoformat() if d['created_at'] else None
            # 判斷發布者是教師還是學生
            if d['teacher_id']:
                d['publisher_id'] = d['teacher_id']
                d['publisher_name'] = d['teacher_username']
                d['is_teacher_post'] = True
            else:
                d['publisher_id'] = d['student_id']
                d['publisher_name'] = d['student_username']
                d['is_teacher_post'] = False

        return jsonify({'success': True, 'discussions': discussions})
    except Exception as e:
        logger.error(f"獲取討論列表失敗: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/discussions', methods=['POST'])
@login_required  # 需要登入才能發討論
def create_discussion():
    try:
        data = request.json
        course_id = data.get('course_id')
        title = data.get('title')
        content = data.get('content')

        if not all([course_id, title, content]):
            return jsonify({'success': False, 'error': '缺少必要資料'}), 400

        connection = get_db_connection()
        cursor = connection.cursor()

        if current_user.role == 'teacher':
            cursor.execute("""
                INSERT INTO discussions (course_id, teacher_id, title, content)
                VALUES (%s, %s, %s, %s)
            """, (course_id, current_user.id, title, content))
        else:
            cursor.execute("""
                INSERT INTO discussions (course_id, student_id, title, content)
                VALUES (%s, %s, %s, %s)
            """, (course_id, current_user.id, title, content))

        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"創建討論失敗: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/discussions/<int:discussion_id>', methods=['DELETE'])
def delete_discussion(discussion_id):
    try:
        user_id = request.json.get('user_id')
        is_teacher = request.json.get('is_teacher', False)

        if not is_teacher:
            return jsonify({'success': False, 'error': '只有老師可以刪除討論'}), 403

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM discussions WHERE discussion_id = %s", (discussion_id,))
        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"刪除討論失敗: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/responses/<int:response_id>', methods=['DELETE'])
def delete_response(response_id):
    try:
        user_id = request.json.get('user_id')
        is_teacher = request.json.get('is_teacher', False)

        if not is_teacher:
            return jsonify({'success': False, 'error': '只有老師可以刪除回覆'}), 403

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM responses WHERE response_id = %s", (response_id,))
        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"刪除回覆失敗: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/responses', methods=['GET'])
def get_responses():
    try:
        discussion_id = request.args.get('discussion_id')
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)  # 使用 dictionary cursor

        cursor.execute("""
            SELECT r.*,
                   CASE 
                     WHEN r.is_teacher = 1 THEN (SELECT username FROM users WHERE user_id = r.user_id)
                     ELSE (SELECT username FROM users WHERE user_id = r.user_id)
                   END as username
            FROM responses r
            WHERE r.discussion_id = %s
            ORDER BY r.created_at ASC
        """, (discussion_id,))

        responses = cursor.fetchall()
        cursor.close()
        connection.close()

        # 確保創建時間的格式正確
        for r in responses:
            r['created_at'] = r['created_at'].isoformat() if r['created_at'] else None

        return jsonify({'success': True, 'responses': responses})
    except Exception as e:
        logger.error(f"獲取回覆失敗: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/responses', methods=['POST'])
def create_response():
    try:
        data = request.json
        discussion_id = data.get('discussion_id')
        user_id = data.get('user_id')
        content = data.get('content')
        is_teacher = data.get('is_teacher', False)

        if not all([discussion_id, user_id, content]):
            return jsonify({'success': False, 'error': '缺少必要資料'}), 400

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO responses (discussion_id, user_id, content, is_teacher)
            VALUES (%s, %s, %s, %s)
        """, (discussion_id, user_id, content, is_teacher))

        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"創建回覆失敗: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/register', methods=['GET'])
def register_page():
    return render_template('register.html')

from flask_bcrypt import Bcrypt

bcrypt = Bcrypt(app)

@app.route('/register', methods=['POST'])
def handle_register():
    try:
        data = request.get_json()
        username = data['username']
        password = data['password']
        role = data['role']

        if user_exists(username):
            return jsonify({'success': False, 'error': '用戶名已存在'}), 409

        # 確保使用 Flask-Bcrypt 來加密密碼
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        create_user(username, hashed_password, role)
        return jsonify({'success': True, 'message': '註冊成功'}), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()

    if user and bcrypt.check_password_hash(user["password_hash"], password):
        user_obj = User(user["user_id"], user["username"], user["role"])
        login_user(user_obj)

        # 🔹 如果 `next` 參數存在，跳轉回原本的頁面，否則跳轉到首頁
        next_page = request.args.get('next')
        return jsonify({'success': True, 'message': '登入成功', 'role': user["role"], 'next': next_page or '/'})
    else:
        return jsonify({'success': False, 'error': '帳號或密碼錯誤'})

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))  # 確保'index'是你的首頁路由名稱




if __name__ == '__main__':
    setup_gpu()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs('static/models', exist_ok=True)
    while not frame_buffer.empty():
        frame_buffer.get()
    while not processed_frame_buffer.empty():
        processed_frame_buffer.get()
    threading.Thread(target=check_thread_status, daemon=True, name="ThreadMonitor").start()
    app.logger.info("🚀 Flask 伺服器啟動: http://127.0.0.1:5000")
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
