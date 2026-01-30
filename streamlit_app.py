import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import numpy as np
import sys
import mediapipe as mp

# MEDIA PIPE FIX FOR PYTHON 3.13+
# MediaPipe currently has issues on Python 3.13. We need to check and warn.
IS_PY313 = sys.version_info >= (3, 13)

try:
    import mediapipe.solutions.face_mesh as mp_face_mesh
    import mediapipe.solutions.drawing_utils as mp_drawing
except (ImportError, AttributeError, ModuleNotFoundError):
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
        from mediapipe.python.solutions import drawing_utils as mp_drawing
    except:
        mp_face_mesh = None
        mp_drawing = None

from collections import deque
import av

# Import functions from main.py if possible, but for a standalone streamlit app, 
# it's often better to have the core logic here or in a shared utils file.
# Since main.py is quite large and intermingled with cv2.imshow, I'll extract and adapt.

class SmileVideoProcessor(VideoProcessorBase):
    def __init__(self):
        if mp_face_mesh is None:
            st.error("CRITICAL: MediaPipe could not be loaded. This is usually due to Python 3.13 incompatibility.")
            st.info("💡 FIX: Go to Streamlit Cloud Settings -> Python version -> Select 3.12")
            return

        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking states
        self.smile_scores = deque(maxlen=10)
        self.baseline_width = None
        self.baseline_widths = deque(maxlen=30)
        self.baseline_eye_aperture = None
        self.baseline_eye_apertures = deque(maxlen=30)
        self.baseline_mouth_height = None
        self.baseline_mouth_heights = deque(maxlen=30)
        
        # Configuration (could be passed from sidebar)
        self.smile_threshold = 0.5
        self.region_size_ratio = 0.65

    def calculate_face_size(self, face_landmarks, image_width, image_height):
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_OUTER = 263
        left_eye_outer = face_landmarks.landmark[LEFT_EYE_OUTER]
        right_eye_outer = face_landmarks.landmark[RIGHT_EYE_OUTER]
        left_eye_outer_px = np.array([left_eye_outer.x * image_width, left_eye_outer.y * image_height])
        right_eye_outer_px = np.array([right_eye_outer.x * image_width, right_eye_outer.y * image_height])
        return np.linalg.norm(right_eye_outer_px - left_eye_outer_px)

    def calculate_eye_aperture(self, face_landmarks, image_width, image_height, face_size):
        LEFT_EYE_TOP = 159
        LEFT_EYE_BOTTOM = 145
        RIGHT_EYE_TOP = 386
        RIGHT_EYE_BOTTOM = 374
        
        left_eye_top = face_landmarks.landmark[LEFT_EYE_TOP]
        left_eye_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM]
        right_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
        right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM]
        
        left_eye_top_px = np.array([left_eye_top.x * image_width, left_eye_top.y * image_height])
        left_eye_bottom_px = np.array([left_eye_bottom.x * image_width, left_eye_bottom.y * image_height])
        right_eye_top_px = np.array([right_eye_top.x * image_width, right_eye_top.y * image_height])
        right_eye_bottom_px = np.array([right_eye_bottom.x * image_width, right_eye_bottom.y * image_height])
        
        left_eye_aperture = np.linalg.norm(left_eye_bottom_px - left_eye_top_px)
        right_eye_aperture = np.linalg.norm(right_eye_bottom_px - right_eye_top_px)
        average_aperture = (left_eye_aperture + right_eye_aperture) / 2.0
        normalized_aperture = average_aperture / face_size if face_size > 0 else average_aperture
        
        return average_aperture, normalized_aperture

    def calculate_smile_score(self, face_landmarks, image_width, image_height, face_size):
        LEFT_MOUTH = 61
        RIGHT_MOUTH = 291
        TOP_LIP_CENTER = 13
        BOTTOM_LIP_CENTER = 14
        
        left_mouth = face_landmarks.landmark[LEFT_MOUTH]
        right_mouth = face_landmarks.landmark[RIGHT_MOUTH]
        top_lip = face_landmarks.landmark[TOP_LIP_CENTER]
        bottom_lip = face_landmarks.landmark[BOTTOM_LIP_CENTER]
        
        left_mouth_px = np.array([left_mouth.x * image_width, left_mouth.y * image_height])
        right_mouth_px = np.array([right_mouth.x * image_width, right_mouth.y * image_height])
        top_lip_px = np.array([top_lip.x * image_width, top_lip.y * image_height])
        bottom_lip_px = np.array([bottom_lip.x * image_width, bottom_lip.y * image_height])
        
        mouth_width = np.linalg.norm(right_mouth_px - left_mouth_px)
        mouth_height = abs(bottom_lip_px[1] - top_lip_px[1])
        
        normalized_mouth_width = mouth_width / face_size if face_size > 0 else mouth_width
        normalized_mouth_height = mouth_height / face_size if face_size > 0 else mouth_height
        
        smile_score = 0.0
        if self.baseline_width is not None and self.baseline_width > 0:
            width_increase = normalized_mouth_width - self.baseline_width
            width_increase_ratio = width_increase / self.baseline_width
            
            SMILE_THRESHOLD = 0.12
            if width_increase_ratio < 0:
                smile_score = 0.0
            elif width_increase_ratio < SMILE_THRESHOLD:
                smile_score = (width_increase_ratio / SMILE_THRESHOLD) * 0.5
            else:
                excess = width_increase_ratio - SMILE_THRESHOLD
                smile_score = 0.5 + min(excess / SMILE_THRESHOLD, 1.0) * 0.5
        
        return min(max(smile_score, 0.0), 1.0), normalized_mouth_width, normalized_mouth_height

    def is_face_front_facing(self, face_landmarks, image_width, image_height):
        # Key landmarks for angle detection
        FOREHEAD = 10
        NOSE_TIP = 4
        CHIN = 175
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_OUTER = 263
        
        forehead = face_landmarks.landmark[FOREHEAD]
        nose_tip = face_landmarks.landmark[NOSE_TIP]
        chin = face_landmarks.landmark[CHIN]
        left_eye = face_landmarks.landmark[LEFT_EYE_OUTER]
        right_eye = face_landmarks.landmark[RIGHT_EYE_OUTER]
        
        forehead_px = np.array([forehead.x * image_width, forehead.y * image_height])
        nose_tip_px = np.array([nose_tip.x * image_width, nose_tip.y * image_height])
        chin_px = np.array([chin.x * image_width, chin.y * image_height])
        left_eye_px = np.array([left_eye.x * image_width, left_eye.y * image_height])
        right_eye_px = np.array([right_eye.x * image_width, right_eye.y * image_height])
        
        face_vertical_length = np.linalg.norm(chin_px - forehead_px)
        if face_vertical_length == 0: return False
        
        face_center_y = (forehead_px[1] + chin_px[1]) / 2
        nose_offset_y = nose_tip_px[1] - face_center_y
        normalized_nose_offset = nose_offset_y / face_vertical_length
        
        eye_vertical_diff = abs(left_eye_px[1] - right_eye_px[1])
        eye_horizontal_dist = abs(right_eye_px[0] - left_eye_px[0])
        eye_alignment_ratio = eye_vertical_diff / eye_horizontal_dist if eye_horizontal_dist > 0 else 0
        
        MAX_NOSE_OFFSET = 0.12
        MAX_EYE_TILT = 0.08
        
        return abs(normalized_nose_offset) <= MAX_NOSE_OFFSET and eye_alignment_ratio <= MAX_EYE_TILT

    def is_face_in_region(self, face_landmarks, image_width, image_height, region):
        rx, ry, rw, rh = region
        region_center_x = rx + rw // 2
        region_center_y = ry + rh // 2
        
        x_coords = [lm.x * image_width for lm in face_landmarks.landmark]
        y_coords = [lm.y * image_height for lm in face_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        face_center_x = (x_min + x_max) // 2
        face_center_y = (y_min + y_max) // 2
        
        return abs(face_center_x - region_center_x) <= rw * 0.25 and abs(face_center_y - region_center_y) <= rh * 0.25

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Mirror effect
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        # Detection Region
        rw = int(w * self.region_size_ratio)
        rh = int(h * self.region_size_ratio)
        rx = (w - rw) // 2
        ry = (h - rh) // 2
        region = (rx, ry, rw, rh)
        
        face_detected = False
        face_landmarks = None
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                if self.is_face_in_region(landmarks, w, h, region) and self.is_face_front_facing(landmarks, w, h):
                    face_detected = True
                    face_landmarks = landmarks
                    break
        
        # Render Guide
        region_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), region_color, 2)
        
        if face_detected:
            face_size = self.calculate_face_size(face_landmarks, w, h)
            smile_score, norm_w, norm_h = self.calculate_smile_score(face_landmarks, w, h, face_size)
            _, norm_eye_ap = self.calculate_eye_aperture(face_landmarks, w, h, face_size)
            
            # Update Baselines
            if smile_score < 0.4:
                self.baseline_widths.append(norm_w)
                if len(self.baseline_widths) > 10: self.baseline_width = sum(self.baseline_widths)/len(self.baseline_widths)
                
                self.baseline_eye_apertures.append(norm_eye_ap)
                if len(self.baseline_eye_apertures) > 10: self.baseline_eye_aperture = sum(self.baseline_eye_apertures)/len(self.baseline_eye_apertures)
                
                self.baseline_mouth_heights.append(norm_h)
                if len(self.baseline_mouth_heights) > 10: self.baseline_mouth_height = sum(self.baseline_mouth_heights)/len(self.baseline_mouth_heights)
            
            # Classification
            status_text = "No Smile"
            status_color = (0, 0, 255)
            
            if smile_score >= 0.5:
                # Laugh detection
                mouth_opening_ratio = norm_h / self.baseline_mouth_height if self.baseline_mouth_height else 1.0
                eye_decrease = ((self.baseline_eye_aperture - norm_eye_ap) / self.baseline_eye_aperture) * 100 if self.baseline_eye_aperture else 0
                
                if mouth_opening_ratio >= 1.35:
                    if eye_decrease >= 12.0:
                        status_text = "Genuine Laugh!"
                        status_color = (255, 0, 255)
                    elif eye_decrease < 3.0:
                        status_text = "Fake Laugh"
                        status_color = (0, 100, 255)
                    else:
                        status_text = "Laugh (Analyzing...)"
                        status_color = (255, 100, 255)
                else:
                    if eye_decrease >= 6.0:
                        status_text = "Genuine Smile!"
                        status_color = (0, 255, 0)
                    elif eye_decrease < 2.0:
                        status_text = "Fake Smile"
                        status_color = (0, 165, 255)
                    else:
                        status_text = "Smile (Analyzing...)"
                        status_color = (0, 255, 255)
            
            cv2.putText(img, status_text, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Simple Smoothing for score display
            self.smile_scores.append(smile_score)
            avg_score = sum(self.smile_scores) / len(self.smile_scores)
            cv2.putText(img, f"Smile Score: {avg_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            cv2.putText(img, "Please Center Your Face", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="Smile Detector", layout="wide")
st.title("😊 Smile Detector")
st.markdown("""
    Welcome to the Smile Detector! This application uses MediaPipe Face Mesh to detect and classify your smiles in real-time.
    
    ### How to use:
    1. Center your face in the green region.
    2. Look directly at the camera.
    3. Flash a smile and see the classification!
""")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="smile-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=SmileVideoProcessor,
    async_processing=True,
)

if webrtc_ctx.video_processor:
    st.sidebar.title("Settings")
    webrtc_ctx.video_processor.region_size_ratio = st.sidebar.slider(
        "Detection Region Size", 0.4, 0.9, 0.65
    )
