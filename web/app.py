from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
import logging
from ultralytics import YOLO
import mediapipe as mp
from werkzeug.utils import secure_filename

app = Flask(__name__,static_folder='static')
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
    'shoulder-press': 'D:\\project_Main\\modles\\yolov8_shoulder_model\\weights\\best.pt'
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load models
models = {}
for exercise_type, model_path in MODEL_PATHS.items():
    try:
        models[exercise_type] = YOLO(model_path)
        logger.info(f"YOLO model for {exercise_type} loaded successfully")
    except Exception as e:
        logger.error(f"Error loading YOLO model for {exercise_type}: {e}")


def clean_detection_info(detection_info):
    """Clean detection info for JSON serialization"""
    cleaned_info = []
    for frame in detection_info:
        cleaned_frame = {
            "angles": {k: float(v) if isinstance(v, (int, float)) else None
                       for k, v in frame["angles"].items()}
        }
        cleaned_info.append(cleaned_frame)
    return cleaned_info


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


# 修改 get_exercise_angles 函數，改為計算所有角度
def get_exercise_angles(landmarks):
    """Calculate all major body angles"""
    angles = {}

    try:
        # 左手肘角度
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angles['左手肘'] = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # 右手肘角度
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angles['右手肘'] = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # 左膝蓋角度
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        angles['左膝蓋'] = calculate_angle(left_hip, left_knee, left_ankle)

        # 右膝蓋角度
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        angles['右膝蓋'] = calculate_angle(right_hip, right_knee, right_ankle)

        # 左肩膀角度
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        angles['左肩膀'] = calculate_angle(left_hip, left_shoulder, left_elbow)

        # 右肩膀角度
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        angles['右肩膀'] = calculate_angle(right_hip, right_shoulder, right_elbow)

        # 左髖部角度
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
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

    # 修改 process_video 函數中調用 get_exercise_angles 的部分
    # 在 process_video 函數中找到這段代碼：
    # Get angles if pose is detected
    frame_angles = {}
    if results_pose.pose_landmarks:
        frame_angles = get_exercise_angles(results_pose.pose_landmarks.landmark)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


def process_video(input_video_path, output_video_path, exercise_type):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Use H.264 codec for better web browser compatibility
    temp_output_path = output_video_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))

    if not out.isOpened():
        # Fallback to XVID codec if H.264 is not available
        out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
        if not out.isOpened():
            raise Exception("Could not create video writer")

    detection_info = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                results_pose = pose.process(frame_rgb)

                # Get angles if pose is detected
                frame_angles = {}
                if results_pose.pose_landmarks:
                    # Updated: Remove exercise_type parameter
                    frame_angles = get_exercise_angles(results_pose.pose_landmarks.landmark)

                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        results_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # Run YOLO model
                model = models[exercise_type]
                results_yolo = model(frame)
                annotated_frame = results_yolo[0].plot()

                # Store detection data
                detection_info.append({"angles": frame_angles})

                out.write(annotated_frame)

        finally:
            cap.release()
            out.release()

        # Convert video to web-compatible format using FFmpeg
        try:
            import subprocess
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_output_path,
                '-vcodec', 'libx264', '-acodec', 'aac',
                output_video_path
            ], check=True)
            os.remove(temp_output_path)  # Remove temporary file
        except Exception as e:
            logger.error(f"Error converting video: {e}")
            # If FFmpeg fails, try to use the temporary file
            if os.path.exists(temp_output_path):
                os.rename(temp_output_path, output_video_path)

    return clean_detection_info(detection_info)


def get_exercise_angles(landmarks):
    """Calculate all major body angles"""
    angles = {}

    try:
        # 左手肘角度
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angles['左手肘'] = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # 右手肘角度
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angles['右手肘'] = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # 左膝蓋角度
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        angles['左膝蓋'] = calculate_angle(left_hip, left_knee, left_ankle)

        # 右膝蓋角度
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        angles['右膝蓋'] = calculate_angle(right_hip, right_knee, right_ankle)

        # 左肩膀角度
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        angles['左肩膀'] = calculate_angle(left_hip, left_shoulder, left_elbow)

        # 右肩膀角度
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        angles['右肩膀'] = calculate_angle(right_hip, right_shoulder, right_elbow)

        # 左髖部角度
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
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


@app.route('/')
def index():
    return render_template('index.html', detection_info=[])


@app.route('/upload', methods=['GET', 'POST'])
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

                detection_info = process_video(input_path, output_path, exercise_type)

                if not os.path.exists(output_path):
                    logger.error(f"Output file not found: {output_path}")
                    return "Error: Processed video not found", 500

                return render_template('uploaded.html',
                                       filename=output_filename,
                                       detection_info=detection_info)

            except Exception as e:
                logger.error(f"Error processing video: {e}")
                return f"Error processing video: {str(e)}", 500

        else:
            logger.warning("Invalid file type")
            return "Invalid file type", 400

    return render_template('index.html', detection_info=[])


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)