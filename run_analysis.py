import os
import numpy as np
import cv2
from collections import Counter, deque
from tensorflow.keras.models import load_model
import dlib
from filterpy.kalman import KalmanFilter
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.backends.opencv import VideoStreamCv2


def run_analysis(input_video_path, output_dir="static/shorts_output"):
    os.makedirs(output_dir, exist_ok=True)

    shape_x, shape_y = 64, 64
    emotion_weights = {"Surprise": 5, "Happy": 4, "Sad": 3, "Angry": 2, "Neutral": 1}
    mapped_emotions = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]

    model = load_model("face_emotion.h5", compile=False)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def init_kalman_filter():
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.R *= 10
        kf.Q *= 0.1
        kf.P *= 100
        return kf

    def detect_face(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60, 60))
        return gray, faces

    def extract_face(gray, faces):
        for x, y, w, h in faces:
            face = gray[y:y+h, x:x+w]
            resized = cv2.resize(face, (shape_x, shape_y)).astype(np.float32) / 255.0
            return np.reshape(resized, (1, shape_x, shape_y, 1))
        return None

    def get_eye_center(eye_points, gray):
        x, y = zip(*eye_points)
        roi = gray[min(y):max(y), min(x):max(x)]
        if roi.size == 0: return (min(x), min(y))
        _, thresh = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(max(contours, key=cv2.contourArea))
            if M['m00'] != 0:
                return (int(M['m10']/M['m00']) + min(x), int(M['m01']/M['m00']) + min(y))
        return (min(x)+(max(x)-min(x))//2, min(y)+(max(y)-min(y))//2)

    def classify_movement(prev, curr):
        if not prev or not curr: return "UNKNOWN"
        dist = np.linalg.norm(np.array(curr) - np.array(prev))
        return "HIGH_FOCUS" if dist < 8 else "MEDIUM_FOCUS" if dist < 20 else "LOW_FOCUS"

    def calculate_attention_score(movement, emotion_score):
        eye_score = 10 if movement == "HIGH_FOCUS" else 5 if movement == "MEDIUM_FOCUS" else 0
        return (eye_score * 0.6) + (emotion_score * 0.4)

    def emotion_weight(emotion):
        return emotion_weights.get(emotion, 0)

    def detect_scenes(video_path):
        video = VideoStreamCv2(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        scene_manager.detect_scenes(video)
        return [(start.get_seconds(), end.get_seconds()) for start, end in scene_manager.get_scene_list()]

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    webcam = cv2.VideoCapture(0)
    kf_left = init_kalman_filter()
    prev_left_pupil = None
    emotion_history = deque(maxlen=5)
    previous_emotion = None
    scene_data = []

    scene_list = detect_scenes(input_video_path)

    for (start, end) in scene_list:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        attention_scores, emotion_scores, emotions = [], [], []
        last_analysis = 0

        while cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0 < end:
            ret_vid, frame = cap.read()
            ret_web, cam = webcam.read()
            if not ret_vid or not ret_web:
                break

            gray, faces = detect_face(cam)
            if len(faces) == 0: continue
            face_img = extract_face(gray, faces)
            if face_img is None: continue

            pred = model.predict(face_img, verbose=0)[0]
            pred[0] += pred[1] + pred[2]; pred[1] = pred[2] = 0
            vec = np.array([pred[0], pred[3], pred[4], pred[5], pred[6]])
            smoothed = np.mean([*emotion_history, vec], axis=0)
            idx = np.argmax(smoothed); emotion = mapped_emotions[idx]
            conf = smoothed[idx]
            emotion_score = 2 if conf > 0.7 else 1 if conf > 0.5 else 0
            if previous_emotion and previous_emotion != emotion:
                emotion_score += 3
            emotion_score += emotion_weight(emotion); emotion_score = min(10, emotion_score)
            previous_emotion = emotion
            emotion_history.append(smoothed)

            dets = detector(gray)
            if len(dets) == 0: continue
            shape = predictor(gray, dets[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye = landmarks[36:42]
            left_pupil = get_eye_center(left_eye, gray)
            movement = classify_movement(prev_left_pupil, left_pupil)
            attention_score = calculate_attention_score(movement, emotion_score)

            attention_scores.append(attention_score)
            emotion_scores.append(emotion_score)
            emotions.append(emotion)
            prev_left_pupil = left_pupil

            # 🎯 오버레이 추가
            cv2.putText(frame, f"{emotion} | Focus: {attention_score:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        scene_data.append({"start": start, "end": end, "attention_scores": attention_scores, "emotion_scores": emotion_scores, "emotions": emotions})

    def calculate_priority(scene):
        focus_ratio = np.mean([1 if s >= 8 else 0.5 if s >= 5 else 0 for s in scene["attention_scores"]]) if scene["attention_scores"] else 0
        emotion_intensity = np.mean(scene["emotion_scores"]) if scene["emotion_scores"] else 0
        dominant_emotion = Counter(scene["emotions"]).most_common(1)[0][0] if scene["emotions"] else "Neutral"
        score = (focus_ratio * 0.6 + (emotion_intensity / 10) * 0.4) * 100
        score += emotion_weight(dominant_emotion) * 3
        return score

    sorted_scenes = sorted(scene_data, key=calculate_priority, reverse=True)

    for i, scene in enumerate(sorted_scenes[:5]):
        start_time = max(scene["start"] - 1.0, 0)
        end_time = scene["end"] + 1.0
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out_path = os.path.join(output_dir, f"short_{i+1:02d}_{scene['start']:.1f}-{scene['end']:.1f}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)

        out.release()

    cap.release()
    webcam.release()
    cv2.destroyAllWindows()
    print("🎉 감정 분석 기반 숏폼 생성 완료")
