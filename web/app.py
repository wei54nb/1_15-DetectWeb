from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
import logging
from ultralytics import YOLO
import mediapipe as mp
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as T
import subprocess

app = Flask(__name__, static_folder='static')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

MODEL_PATHS = {
    'squat': 'D:\\project_Main\\modles\\yolov8_squat_model\\weights\\best.pt',
    'bicep-curl': 'D:\\project_Main\\modles\\yolov8_bicep_model2\\weights\\best.pt',
    'shoulder-press': 'D:\\project_Main\\modles\\yolov8_shoulder_model\\weights\\best.pt',
    'push-up':'D:\\project_Main\\modles\\push-up_model\\weights\\pushup_best.pt'
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load models
models = {}
for exercise_type, model_path in MODEL_PATHS.items():
    try:
        models[exercise_type] = YOLO(model_path).to('cuda')
        logger.info(f"YOLO model for {exercise_type} loaded successfully on GPU")
    except Exception as e:
        logger.error(f"Error loading YOLO model for {exercise_type}: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_exercise_angles(landmarks):
    """Calculate all major body angles"""
    angles = {}

    try:
        # Calculate angles for major body joints
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
                min_tracking_confidence=0.5) as pose:

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
                        angles = get_exercise_angles(landmarks)

                        frame_info['angles'] = angles

                    if len(results_yolo[0].boxes) > 0:
                        for box in results_yolo[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf)
                            class_id = int(box.cls)
                            class_name = model.names[class_id]

                            # 繪製邊界框
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
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


@app.route('/')
def index():
    return render_template('index.html')


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
    """處理影片串流"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        logger.info(f"Attempting to stream file: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return "File not found", 404

        # 添加正確的 MIME 類型
        return send_from_directory(
            app.config['OUTPUT_FOLDER'],
            filename,
            mimetype='video/mp4'
        )

    except Exception as e:
        logger.error(f"Error streaming video: {e}")
        return f"Error streaming video: {str(e)}", 500


def partial_video_stream(file_path, start, end):
    """分段讀取視頻文件"""
    with open(file_path, 'rb') as video:
        video.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk_size = min(8192, remaining)  # 8KB chunks
            data = video.read(chunk_size)
            if not data:
                break
            remaining -= len(data)
            yield data


def calculate_exercise_angles_mediapipe(landmarks, exercise_type):
    """統一計算所有運動的關鍵角度（左右兩側）"""
    angles = {}

    try:
        # 右側關鍵點
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
        right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

        # 左側關鍵點
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        left_elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
        left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
        left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])

        # 計算右側角度
        angles['right_shoulder'] = calculate_angle(right_elbow, right_shoulder, right_hip)
        angles['right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
        angles['right_hip'] = calculate_angle(right_shoulder, right_hip, right_knee)
        angles['right_knee'] = calculate_angle(right_hip, right_knee, right_ankle)

        # 計算左側角度
        angles['left_shoulder'] = calculate_angle(left_elbow, left_shoulder, left_hip)
        angles['left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
        angles['left_hip'] = calculate_angle(left_shoulder, left_hip, left_knee)
        angles['left_knee'] = calculate_angle(left_hip, left_knee, left_ankle)

    except Exception as e:
        logger.error(f"Error calculating angles: {e}")

    return angles


def calculate_angle(p1, p2, p3):
    """计算三个点形成的角度"""
    # 将点转换为numpy数组
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # 计算两个向量
    v1 = p1 - p2
    v2 = p3 - p2

    # 计算角度（弧度）
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    # 转换为角度
    return np.degrees(angle)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)