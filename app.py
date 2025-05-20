from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import difflib
import csv
from moviepy import VideoFileClip
import speech_recognition as sr
import traceback
import pandas as pd
import random

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploaded_videos'
LANDMARKS_DIR = './guide_landmarks/guide_landmarks'
SYLLABLE_LANDMARKS_DIR = './guide_landmarks/발음녹화_음절'
EXCEL_PATH = './guide_landmarks/letters_for_letters.xlsx'
syllable_book = pd.read_excel(EXCEL_PATH, sheet_name=None)  # 모든 시트를 딕셔너리로 불러옴
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LIPS_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
            317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

current_signal = None


def load_landmarks(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def compare_lip_shape(user_data, ref_data):
    user_array = np.array(user_data)  # (Nframes_user, 80)
    user_array = user_array.reshape(-1, 40, 2)
    avg_user = np.mean(user_array, axis=0)  # (40, 2)
    ref_points = np.array(ref_data)  # 이미 (40, 2)
    distances = np.linalg.norm(avg_user - ref_points, axis=1)
    return float(np.mean(distances))

def give_lip_feedback(distance):
    if distance < 0.13:
        return "정확해요!"
    elif distance < 0.21:
        return "나쁘지 않아요!"
    else:
        return "입모양이 많이 달라요."

def load_landmarks_from_csv(filepath):
    landmarks = []
    try:
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                landmarks.append([float(val) for val in row])
        # landmarks는 (Nframes, 80) 개 좌표(x,y*40) 리스트 형태일 것
        arr = np.array(landmarks)  # shape (Nframes, 80)
        if arr.shape[1] != 80:
            print(f"경고: CSV 데이터 열 개수가 예상과 다름: {arr.shape[1]}")
            return None
        arr = arr.reshape(-1, 40, 2)  # (Nframes, 40, 2)
        avg_landmarks = np.mean(arr, axis=0)  # (40, 2)
        return avg_landmarks.tolist()
    except Exception as e:
        print(f"CSV 로딩 실패: {e}")
        return None

def load_reference_phoneme(syllable):
    path = os.path.join(SYLLABLE_LANDMARKS_DIR, f"좌표_{syllable}_avg.csv")
    return load_landmarks_from_csv(path)

def analyze_syllables(user_text, current_data):
    syllables = list(user_text.replace(" ", ""))
    if not syllables:
        return "인식된 음절이 없습니다."

    total_frames = len(current_data)
    total_syllables = len(syllables)
    if total_syllables == 1:
        syllable_frames = [current_data]
    else:
        frames_per_syllable = total_frames // total_syllables
        syllable_frames = [
            current_data[i * frames_per_syllable : (i + 1) * frames_per_syllable]
            for i in range(total_syllables)
        ]

    lip_msg = "\n[입모양 분석]\n"
    for idx, syllable in enumerate(syllables):
        syllable_lip_data = syllable_frames[idx] if idx < len(syllable_frames) else []
        ref_data = load_reference_phoneme(syllable)
        if ref_data:
            if syllable_lip_data:
                dist = compare_lip_shape(syllable_lip_data, ref_data)
                feedback = give_lip_feedback(dist)
                lip_msg += f"- '{syllable}' → 거리: {dist:.4f}, 피드백: {feedback}\n"
            else:
                lip_msg += f"- '{syllable}' → 사용자 데이터 부족\n"
        else:
            lip_msg += f"- '{syllable}' → 기준 데이터 없음\n"
    return lip_msg

def get_word_from_initials(initials):
    if initials not in syllable_book:
        return None
    words = syllable_book[initials].dropna().values.flatten().tolist()
    return random.choice(words) if words else None

def extract_text_from_video(video_path):
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            real_sound = recognizer.recognize_google(audio, language="ko-KR")
    except Exception as e:
        messagebox.showerror("음성 인식 오류", f"오류: {e}")
        return

    status_label.config(text=f"인식된 문장: {user_text}")


def analyze_word_similarity(user_text, target_text):
    user_words = user_text.split()
    target_words = target_text.split()
    used_indices = set()
    word_feedback = []

    for idx, t_word in enumerate(target_words):
        best_match = None
        best_score = 0
        best_idx = -1
        for i, u_word in enumerate(user_words):
            if i in used_indices:
                continue
            score = difflib.SequenceMatcher(None, t_word, u_word).ratio()
            if score > best_score:
                best_score = score
                best_match = u_word
                best_idx = i

        if best_match is not None:
            used_indices.add(best_idx)
            pos = "위치 일치" if best_idx == idx else "위치 바뀜"

            # 🔍 상세 피드백 생성
            hint = ""
            if t_word != best_match:
                if t_word.startswith(best_match):
                    hint = " → 끝 소리가 빠졌을 수 있어요."
                elif best_match.startswith(t_word):
                    hint = " → 끝에 불필요한 소리가 붙었을 수 있어요."
                elif sorted(t_word) == sorted(best_match):
                    hint = " → 철자는 비슷하지만 순서가 바뀐 것 같아요."
                else:
                    hint = " → 발음이 비슷하지만 다른 단어일 수 있어요."

            word_feedback.append(
                f"- '{t_word}' ↔ '{best_match}' : {best_score*100:.1f}% ({pos}){hint}"
            )
        else:
            word_feedback.append(f"- '{t_word}' ↔ (누락됨) : 0.0%")

    extra_words = [w for i, w in enumerate(user_words) if i not in used_indices]
    if extra_words:
        word_feedback.append(f"\n추가로 말한 단어들: {' '.join(extra_words)}")

    return word_feedback

def analyze_lip_movement(video_path, expected_mouth):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': '비디오 파일을 열 수 없습니다.'}

    lip_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            lip_coords = [coord for idx in LIPS_IDX for coord in (face.landmark[idx].x, face.landmark[idx].y)]
            lip_data.append(lip_coords)
    cap.release()

    standard_path = os.path.join(LANDMARKS_DIR, f"guide_landmarks_{expected_mouth}.json")
    print(f"표준 경로: {standard_path}")
    if not os.path.exists(standard_path):
        return {'error': f"표준 파일 '{standard_path}' 이 존재하지 않습니다."}

    standard_data = load_landmarks(standard_path)
    if not standard_data:
        return {'error': f"파일 열기 실패: '{standard_path}'"}
    if 'lips' not in standard_data:
        return {'error': f"'lips' 키가 '{standard_path}'에 없습니다."}

    standard_lips = standard_data['lips']
    if len(standard_lips) != len(LIPS_IDX):
        return {'error': f"표준 입술 랜드마크 개수가 맞지 않습니다."}

    total_similarity = 0
    max_distance = 0.1
    for frame_lip_coords in lip_data:
        user_lip_points = np.array(frame_lip_coords).reshape(-1, 2)
        standard_lip_points = np.array([standard_lips[i] for i in range(len(LIPS_IDX))])
        distances = np.linalg.norm(user_lip_points - standard_lip_points, axis=1)
        avg_distance = np.sqrt(np.mean(distances**2))
        if avg_distance > max_distance:
            similarity = 0
        else:
            similarity = 1 - (avg_distance / max_distance) ** 2
        total_similarity += similarity * 100

    average_similarity = total_similarity / len(lip_data)
    return {
    'accuracy': f"{average_similarity:.2f}",
    'lip_data': lip_data  # syllable 분석용
}

@app.route('/signal', methods=['POST'])
def receive_signal():
    global current_signal
    data = request.get_json()
    signal = data.get('signal')
    if signal in [0, 1,2, 3, 4]:
        current_signal = signal
        return jsonify({'message': f'Signal {signal} received'}), 200
    else:
        return jsonify({'error': 'Invalid signal'}), 400

@app.route('/upload_and_analyze', methods=['POST'])
def upload_and_analyze():
    global current_signal
    if current_signal != 2:
        return jsonify({'error': 'Signal 2가 필요합니다.'}), 403
    current_signal = None
    file = request.files.get('file')
    expected_mouth = request.form.get('mouth')
    if not file or not expected_mouth:
        return jsonify({'error': '파일 또는 mouth 정보가 없습니다.'}), 400

    filename = f"uploaded_{expected_mouth}_{int(time.time())}.mp4"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    result = analyze_lip_movement(save_path, expected_mouth)
    print("최종 분석 메시지:\n", result)
    os.remove(save_path)
    return jsonify(result)
    
@app.route('/get_word', methods=['POST'])
def get_word():
    initials = request.json.get('initials', '')
    if not initials:
        return jsonify({'error': '초성이 필요합니다.'}), 400
    word = get_word_from_initials(initials)
    if not word:
        return jsonify({'error': f"초성 '{initials}'에 해당하는 단어 없음"}), 404
    return jsonify({'word': word})

@app.route('/upload_and_analyze_syllable', methods=['POST'])
def upload_and_analyze_syllable():
    global current_signal
    if current_signal != 3:
        return jsonify({'error': 'Signal 3이 필요합니다.'}), 403
    current_signal = None

    file = request.files.get('file')
    user_text = request.form.get('user_text', '')
    if not file or not user_text:
        return jsonify({'error': '파일 또는 단어 누락'}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = f"syllable_{int(time.time())}.mp4"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # 영상 처리
    cap = cv2.VideoCapture(save_path)
    current_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)  # 미디어파이프 객체가 초기화되어 있어야 함
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            lip_coords = [coord for idx in LIPS_IDX for coord in (face.landmark[idx].x, face.landmark[idx].y)]
            current_data.append(lip_coords)
    cap.release()
    os.remove(save_path)

    syllable_result = analyze_syllables(user_text, current_data)
    return jsonify({
        'syllable_result': syllable_result,
        'selected_word': user_text  # 프론트에 단어 표시를 위한 반환
    })

@app.route('/upload_and_analyze_sentence', methods=['POST'])
def upload_and_analyze_sentence():
    try:
        global current_signal
        if current_signal != 4:
            return jsonify({'error': 'Signal 4가 필요합니다.'}), 403
        current_signal = None

        file = request.files.get('file')
        user_text = request.form.get('user_text', '')  # 사용자가 입력한 정답 문장

        if not file or not user_text:
            return jsonify({'error': '파일 또는 user_text 누락'}), 400

        save_path = 'temp_video.mp4'
        wav_path = 'temp_audio.wav'
        file.save(save_path)

        # 오디오 추출
        clip = VideoFileClip(save_path)
        clip.audio.write_audiofile(wav_path)
        clip.reader.close()
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            realsound = recognizer.recognize_google(audio, language='ko-KR')
        os.remove(wav_path)

        ########## 전체 문장 발음 정확도 ##########
        accuracy = difflib.SequenceMatcher(None, realsound, user_text).ratio() * 100
        message = f"전체 문장 발음 정확도: {accuracy:.2f}%\n\n"

        ########## 단어별 피드백 ##########
        real_words = realsound.split()
        target_words = user_text.split()
        used_indices = set()
        word_feedback = []

        for idx, t_word in enumerate(target_words):
            best_match = None
            best_score = 0
            best_idx = -1
            for i, r_word in enumerate(real_words):
                if i in used_indices:
                    continue
                score = difflib.SequenceMatcher(None, t_word, r_word).ratio()
                if score > best_score:
                    best_score = score
                    best_match = r_word
                    best_idx = i
            if best_match is not None:
                used_indices.add(best_idx)
                pos = "위치 일치" if best_idx == idx else "위치 바뀜"
                word_feedback.append(f"- '{t_word}' ↔ '{best_match}' : {best_score*100:.1f}% ({pos})")
            else:
                word_feedback.append(f"- '{t_word}' ↔ (누락됨) : 0.0%")

        extra_words = [w for i, w in enumerate(real_words) if i not in used_indices]
        if extra_words:
            word_feedback.append(f"\n추가로 말한 단어들: {' '.join(extra_words)}")

        message += "[단어별 분석]\n" + "\n".join(word_feedback)

        ########## 입모양 데이터 수집 ##########
        cap = cv2.VideoCapture(save_path)
        current_data = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                lip_coords = [coord for idx in LIPS_IDX for coord in (face.landmark[idx].x, face.landmark[idx].y)]
                current_data.append(lip_coords)
        cap.release()
        os.remove(save_path)

        ########## 음절별 입모양 분석 ##########
        lip_msg = "\n\n[입모양 분석]\n"
        syllables = list(user_text.replace(" ", ""))  # 기준 문장의 음절로 분석

        if not syllables:
            lip_msg += "인식된 음절이 없습니다.\n"
        else:
            total_frames = len(current_data)
            total_syllables = len(syllables)

            if total_syllables == 1:
                syllable_frames = [current_data]
            else:
                frames_per_syllable = total_frames // total_syllables
                syllable_frames = [
                    current_data[i * frames_per_syllable : (i + 1) * frames_per_syllable]
                    for i in range(total_syllables)
                ]

            for idx, syllable in enumerate(syllables):
                syllable_lip_data = syllable_frames[idx] if idx < len(syllable_frames) else []
                ref_data = load_reference_phoneme(syllable)
                if ref_data:
                    if syllable_lip_data:
                        dist = compare_lip_shape(syllable_lip_data, ref_data)
                        feedback = give_lip_feedback(dist)
                        lip_msg += f"- '{syllable}' → 거리: {dist:.4f}, 피드백: {feedback}\n"
                    else:
                        lip_msg += f"- '{syllable}' → 사용자 데이터 부족\n"
                else:
                    lip_msg += f"- '{syllable}' → 기준 데이터 없음\n"

        message += lip_msg if lip_msg else "\n기준 좌표가 없는 음절이 포함되어 있습니다."

        return jsonify({'message': message, 'accuracy': f"{accuracy:.2f}"})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(LANDMARKS_DIR):
        print(f"경고: 기준 데이터 디렉토리 없음 → {LANDMARKS_DIR}")
    else:
        print(f"📁 기준 데이터 디렉토리 확인됨: {LANDMARKS_DIR}")
    port = int(os.environ.get("PORT", 5000))  # Render가 PORT 환경변수를 제공
    app.run(host="0.0.0.0", port=port)
