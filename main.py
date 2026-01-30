"""
Smile Detection using MediaPipe Facial Landmarks
Accurate approach using facial landmark detection
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque


def calculate_eye_aperture(face_landmarks, image_width, image_height):
    """
    Calculate eye aperture (vertical opening) for both eyes
    Genuine smiles cause eyes to become smaller (aperture decreases)
    
    Args:
        face_landmarks: MediaPipe face landmarks
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        tuple: (average_eye_aperture, left_eye_aperture, right_eye_aperture, eye_coords)
    """
    # MediaPipe face mesh eye landmark indices
    # Left eye: top (159), bottom (145)
    # Right eye: top (386), bottom (374)
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    
    # Get eye landmarks
    left_eye_top = face_landmarks.landmark[LEFT_EYE_TOP]
    left_eye_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM]
    right_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
    right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM]
    
    # Convert to pixel coordinates
    left_eye_top_px = np.array([left_eye_top.x * image_width, left_eye_top.y * image_height])
    left_eye_bottom_px = np.array([left_eye_bottom.x * image_width, left_eye_bottom.y * image_height])
    right_eye_top_px = np.array([right_eye_top.x * image_width, right_eye_top.y * image_height])
    right_eye_bottom_px = np.array([right_eye_bottom.x * image_width, right_eye_bottom.y * image_height])
    
    # Calculate vertical distance (aperture) for each eye
    left_eye_aperture = np.linalg.norm(left_eye_bottom_px - left_eye_top_px)
    right_eye_aperture = np.linalg.norm(right_eye_bottom_px - right_eye_top_px)
    
    # Average aperture of both eyes
    average_aperture = (left_eye_aperture + right_eye_aperture) / 2.0
    
    # Store eye coordinates for visualization
    eye_coords = {
        'left_eye_top': (int(left_eye_top_px[0]), int(left_eye_top_px[1])),
        'left_eye_bottom': (int(left_eye_bottom_px[0]), int(left_eye_bottom_px[1])),
        'right_eye_top': (int(right_eye_top_px[0]), int(right_eye_top_px[1])),
        'right_eye_bottom': (int(right_eye_bottom_px[0]), int(right_eye_bottom_px[1]))
    }
    
    return average_aperture, left_eye_aperture, right_eye_aperture, eye_coords


def calculate_smile_score_from_landmarks(face_landmarks, image_width, image_height, baseline_width=None):
    """
    Calculate smile score (0-1) based on mouth width increase from baseline
    
    Args:
        face_landmarks: MediaPipe face landmarks
        image_width: Width of the image
        image_height: Height of the image
        baseline_width: Baseline mouth width when not smiling (None for first frame)
    
    Returns:
        tuple: (smile_score (0-1), lip_coordinates_dict, current_mouth_width)
    """
    # MediaPipe face mesh landmark indices for mouth
    # Full mouth outline landmarks
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    TOP_LIP_CENTER = 13
    BOTTOM_LIP_CENTER = 14
    TOP_LIP_LEFT = 78
    TOP_LIP_RIGHT = 308
    BOTTOM_LIP_LEFT = 95
    BOTTOM_LIP_RIGHT = 325
    
    # Additional mouth landmarks for full outline
    # Outer mouth landmarks (for full lip outline)
    MOUTH_LANDMARKS = [
        61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
        13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324
    ]
    
    # Inner mouth landmarks
    INNER_MOUTH_LANDMARKS = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 13, 82, 81, 80
    ]
    
    # Get landmark points
    left_mouth = face_landmarks.landmark[LEFT_MOUTH]
    right_mouth = face_landmarks.landmark[RIGHT_MOUTH]
    top_lip = face_landmarks.landmark[TOP_LIP_CENTER]
    bottom_lip = face_landmarks.landmark[BOTTOM_LIP_CENTER]
    top_lip_left = face_landmarks.landmark[TOP_LIP_LEFT]
    top_lip_right = face_landmarks.landmark[TOP_LIP_RIGHT]
    bottom_lip_left = face_landmarks.landmark[BOTTOM_LIP_LEFT]
    bottom_lip_right = face_landmarks.landmark[BOTTOM_LIP_RIGHT]
    
    # Convert to pixel coordinates
    left_mouth_px = np.array([left_mouth.x * image_width, left_mouth.y * image_height])
    right_mouth_px = np.array([right_mouth.x * image_width, right_mouth.y * image_height])
    top_lip_px = np.array([top_lip.x * image_width, top_lip.y * image_height])
    bottom_lip_px = np.array([bottom_lip.x * image_width, bottom_lip.y * image_height])
    
    # Calculate mouth width (distance between corners)
    mouth_width = np.linalg.norm(right_mouth_px - left_mouth_px)
    
    # Calculate mouth height (vertical distance between top and bottom lip centers)
    mouth_height = abs(bottom_lip_px[1] - top_lip_px[1])
    
    # Calculate additional lip points for coordinate display
    top_lip_left_px = np.array([top_lip_left.x * image_width, top_lip_left.y * image_height])
    top_lip_right_px = np.array([top_lip_right.x * image_width, top_lip_right.y * image_height])
    bottom_lip_left_px = np.array([bottom_lip_left.x * image_width, bottom_lip_left.y * image_height])
    bottom_lip_right_px = np.array([bottom_lip_right.x * image_width, bottom_lip_right.y * image_height])
    
    # SMILE DETECTION BASED ON MOUTH WIDTH INCREASE (HORIZONTAL LINE LENGTH)
    # When smiling, the horizontal line (mouth width) increases from baseline
    # Calculate smile score based on width increase from baseline
    
    if baseline_width is None or baseline_width == 0:
        # No baseline yet - assume neutral (not smiling)
        smile_score = 0.0
        width_increase_ratio = 0.0
    else:
        # Calculate width increase ratio (percentage increase)
        width_increase = mouth_width - baseline_width
        width_increase_ratio = width_increase / baseline_width if baseline_width > 0 else 0.0
        
        # Threshold: 15% increase = smile threshold
        # Scale: 0% = 0.0, 15% = 0.5, 30%+ = 1.0
        SMILE_THRESHOLD = 0.15  # 15% increase threshold
        
        if width_increase_ratio < 0:
            # Width decreased (mouth closed more) - definitely not smiling
            smile_score = 0.0
        elif width_increase_ratio < SMILE_THRESHOLD:
            # Below threshold - not smiling
            # Linear scale from 0 to threshold (0 to 0.5 score)
            smile_score = (width_increase_ratio / SMILE_THRESHOLD) * 0.5
        else:
            # Above threshold - smiling
            # Scale from threshold to 2x threshold (15% to 30% increase = full smile)
            excess = width_increase_ratio - SMILE_THRESHOLD
            max_excess = SMILE_THRESHOLD  # Another 15% for full score
            smile_score = 0.5 + min(excess / max_excess, 1.0) * 0.5  # 0.5 to 1.0
    
    # Ensure score is between 0 and 1
    smile_score = min(max(smile_score, 0.0), 1.0)
    
    # Get ALL mouth landmark coordinates
    # MediaPipe face mesh has 468 landmarks
    # Mouth region landmarks (outer and inner)
    # These are the standard MediaPipe mouth landmark indices
    mouth_landmark_indices = [
        61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,  # Outer upper
        13, 82, 81, 80, 78,  # Outer lower
        95, 88, 178, 87, 14, 317, 402, 318, 324  # Inner mouth
    ]
    
    # Collect all mouth landmark coordinates
    all_mouth_coords = {}
    
    for idx in mouth_landmark_indices:
        if idx < len(face_landmarks.landmark):
            lm = face_landmarks.landmark[idx]
            all_mouth_coords[f'landmark_{idx}'] = (int(lm.x * image_width), int(lm.y * image_height))
    
    # Key points for easy access
    lip_coords = {
        'left_corner': (int(left_mouth_px[0]), int(left_mouth_px[1])),
        'right_corner': (int(right_mouth_px[0]), int(right_mouth_px[1])),
        'top_center': (int(top_lip_px[0]), int(top_lip_px[1])),
        'bottom_center': (int(bottom_lip_px[0]), int(bottom_lip_px[1])),
        'top_lip_left': (int(top_lip_left_px[0]), int(top_lip_left_px[1])),
        'top_lip_right': (int(top_lip_right_px[0]), int(top_lip_right_px[1])),
        'bottom_lip_left': (int(bottom_lip_left_px[0]), int(bottom_lip_left_px[1])),
        'bottom_lip_right': (int(bottom_lip_right_px[0]), int(bottom_lip_right_px[1])),
        'all_landmarks': all_mouth_coords  # All mouth landmarks
    }
    
    return smile_score, lip_coords, mouth_width


def detect_smile():
    """
    Main function to detect smiles in real-time using webcam with MediaPipe landmarks
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize webcam - auto-detect iPhone/Camo camera
    cap = None
    camera_indices = [1, 2, 0, 3]  # Try iPhone/Camo cameras first (usually 1 or 2), then default
    
    print("Searching for camera...")
    for camera_index in camera_indices:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera found at index: {camera_index}")
                break
            cap.release()
            cap = None
        else:
            if cap:
                cap.release()
            cap = None
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open any camera")
        print("Make sure your iPhone camera (Camo) is running and connected")
        return
    
    print("Smile Detection Started!")
    print("Press 'q' to quit")
    
    # Use a deque for smoothing (moving average)
    smile_scores = deque(maxlen=10)  # Average over last 10 frames
    baseline_width = None  # Baseline mouth width (neutral expression)
    baseline_widths = deque(maxlen=30)  # Track baseline over time
    
    # Eye aperture tracking for genuine/fake smile detection
    baseline_eye_aperture = None  # Baseline eye aperture (neutral expression)
    baseline_eye_apertures = deque(maxlen=30)  # Track baseline eye aperture over time
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = face_mesh.process(frame_rgb)
            
            h, w = frame.shape[:2]
            current_smile_score = 0.0
            smile_text = "No Face"
            smile_color = (128, 128, 128)
            lip_coords = None
            current_mouth_width = 0.0
            current_eye_aperture = 0.0
            eye_coords = None
            smile_type = ""  # "Genuine" or "Fake"
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate smile score from landmarks
                    current_smile_score, lip_coords, current_mouth_width = calculate_smile_score_from_landmarks(
                        face_landmarks, w, h, baseline_width
                    )
                    
                    # Calculate eye aperture
                    avg_aperture, left_aperture, right_aperture, eye_coords = calculate_eye_aperture(
                        face_landmarks, w, h
                    )
                    current_eye_aperture = avg_aperture
                    
                    # Update baseline width when not smiling (score < 0.3)
                    # Use rolling average to establish baseline
                    if current_smile_score < 0.3:
                        baseline_widths.append(current_mouth_width)
                        if len(baseline_widths) > 5:  # After 5 frames, establish baseline
                            baseline_width = sum(baseline_widths) / len(baseline_widths)
                    elif baseline_width is None and len(baseline_widths) > 0:
                        # If no baseline yet, use current width as initial baseline
                        baseline_width = sum(baseline_widths) / len(baseline_widths) if baseline_widths else current_mouth_width
                    
                    # Update baseline eye aperture when not smiling (score < 0.3)
                    if current_smile_score < 0.3:
                        baseline_eye_apertures.append(current_eye_aperture)
                        if len(baseline_eye_apertures) > 5:  # After 5 frames, establish baseline
                            baseline_eye_aperture = sum(baseline_eye_apertures) / len(baseline_eye_apertures)
                    elif baseline_eye_aperture is None and len(baseline_eye_apertures) > 0:
                        # If no baseline yet, use current aperture as initial baseline
                        baseline_eye_aperture = sum(baseline_eye_apertures) / len(baseline_eye_apertures) if baseline_eye_apertures else current_eye_aperture
                    
                    # Draw face landmarks (optional - can comment out)
                    # mp_drawing.draw_landmarks(
                    #     frame,
                    #     face_landmarks,
                    #     mp_face_mesh.FACEMESH_CONTOURS,
                    #     None,
                    #     mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                    # )
                    
                    # Get face bounding box for display
                    x_coords = [lm.x * w for lm in face_landmarks.landmark]
                    y_coords = [lm.y * h for lm in face_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Draw face bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.putText(frame, 'Face', (x_min, y_min - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Draw ALL lip coordinates if detected
                    if lip_coords:
                        left_corner = lip_coords['left_corner']
                        right_corner = lip_coords['right_corner']
                        top_center = lip_coords['top_center']
                        bottom_center = lip_coords['bottom_center']
                        
                        # Draw lip corners (green circles)
                        cv2.circle(frame, left_corner, 5, (0, 255, 0), -1)
                        cv2.circle(frame, right_corner, 5, (0, 255, 0), -1)
                        
                        # Draw line connecting corners
                        cv2.line(frame, left_corner, right_corner, (0, 255, 0), 2)
                        
                        # Draw top and bottom lip centers (blue circles)
                        cv2.circle(frame, top_center, 4, (255, 0, 0), -1)
                        cv2.circle(frame, bottom_center, 4, (255, 0, 0), -1)
                        
                        # Draw additional key lip points (yellow)
                        cv2.circle(frame, lip_coords['top_lip_left'], 3, (0, 255, 255), -1)
                        cv2.circle(frame, lip_coords['top_lip_right'], 3, (0, 255, 255), -1)
                        cv2.circle(frame, lip_coords['bottom_lip_left'], 3, (0, 255, 255), -1)
                        cv2.circle(frame, lip_coords['bottom_lip_right'], 3, (0, 255, 255), -1)
                        
                        # Draw ALL mouth landmarks (small cyan dots)
                        if 'all_landmarks' in lip_coords:
                            for landmark_name, coord in lip_coords['all_landmarks'].items():
                                cv2.circle(frame, coord, 2, (255, 255, 0), -1)
                        
                        # Display key coordinates
                        coord_text = f'L:{left_corner[0]},{left_corner[1]} R:{right_corner[0]},{right_corner[1]}'
                        cv2.putText(frame, coord_text, (x_min, y_max + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Display all coordinates as text (scrollable if needed)
                        all_coords_text = "All Lip Coords: "
                        coord_list = []
                        if 'all_landmarks' in lip_coords:
                            for landmark_name, coord in lip_coords['all_landmarks'].items():
                                coord_list.append(f"{landmark_name}:({coord[0]},{coord[1]})")
                        
                        # Display coordinates in multiple lines if needed
                        coord_display = ", ".join(coord_list[:5])  # Show first 5
                        if len(coord_list) > 5:
                            coord_display += f" ... (+{len(coord_list)-5} more)"
                        cv2.putText(frame, coord_display, (x_min, y_max + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
                    # Binary classification: No Smile or Smile
                    # Threshold based on width increase (0.5 = 15% increase threshold)
                    if current_smile_score >= 0.5:
                        # Smile detected - now check if it's genuine or fake
                        if baseline_eye_aperture and baseline_eye_aperture > 0:
                            # Calculate eye aperture change (percentage decrease)
                            eye_aperture_decrease = ((baseline_eye_aperture - current_eye_aperture) / baseline_eye_aperture) * 100
                            
                            # Genuine smile: eyes become smaller (aperture decreases by at least 5%)
                            # Fake smile: eyes don't become smaller (aperture stays same or increases)
                            if eye_aperture_decrease >= 5.0:  # 5% decrease threshold
                                smile_text = 'Genuine Smile!'
                                smile_color = (0, 255, 0)  # Green
                                smile_type = "Genuine"
                            else:
                                smile_text = 'Fake Smile'
                                smile_color = (0, 165, 255)  # Orange
                                smile_type = "Fake"
                        else:
                            # Baseline not established yet
                            smile_text = 'Smile (Analyzing...)'
                            smile_color = (0, 255, 255)  # Yellow
                            smile_type = "Unknown"
                    else:
                        smile_text = 'No Smile'
                        smile_color = (0, 0, 255)  # Red
                        smile_type = ""
                    
                    # Display smile status
                    cv2.putText(frame, smile_text, (x_min, y_min - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, smile_color, 2)
                    
                    # Draw eye landmarks for visualization
                    if eye_coords:
                        # Draw left eye (top and bottom)
                        cv2.circle(frame, eye_coords['left_eye_top'], 4, (255, 255, 0), -1)
                        cv2.circle(frame, eye_coords['left_eye_bottom'], 4, (255, 255, 0), -1)
                        cv2.line(frame, eye_coords['left_eye_top'], eye_coords['left_eye_bottom'], (255, 255, 0), 2)
                        
                        # Draw right eye (top and bottom)
                        cv2.circle(frame, eye_coords['right_eye_top'], 4, (255, 255, 0), -1)
                        cv2.circle(frame, eye_coords['right_eye_bottom'], 4, (255, 255, 0), -1)
                        cv2.line(frame, eye_coords['right_eye_top'], eye_coords['right_eye_bottom'], (255, 255, 0), 2)
                    
                    # Display mouth width information
                    if baseline_width and baseline_width > 0:
                        width_increase = ((current_mouth_width - baseline_width) / baseline_width) * 100
                        width_text = f'Width: {current_mouth_width:.1f}px (Baseline: {baseline_width:.1f}px, +{width_increase:.1f}%)'
                        cv2.putText(frame, width_text, (x_min, y_max + 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display eye aperture information
                    if baseline_eye_aperture and baseline_eye_aperture > 0:
                        eye_decrease = ((baseline_eye_aperture - current_eye_aperture) / baseline_eye_aperture) * 100
                        eye_text = f'Eye Aperture: {current_eye_aperture:.1f}px (Baseline: {baseline_eye_aperture:.1f}px, {eye_decrease:+.1f}%)'
                        eye_color = (0, 255, 0) if eye_decrease >= 5.0 else (0, 165, 255) if current_smile_score >= 0.5 else (255, 255, 255)
                        cv2.putText(frame, eye_text, (x_min, y_max + 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)
            
            # Add current score to smoothing buffer
            smile_scores.append(current_smile_score)
            
            # Calculate smoothed average score
            smoothed_score = sum(smile_scores) / len(smile_scores) if smile_scores else 0.0
            
            # Display statistics with 0-1 range
            cv2.putText(frame, f'Smile Score: {smoothed_score:.3f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Current: {current_smile_score:.3f}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            if baseline_width:
                cv2.putText(frame, f'Baseline Width: {baseline_width:.1f}px', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            if baseline_eye_aperture:
                cv2.putText(frame, f'Baseline Eye Aperture: {baseline_eye_aperture:.1f}px', (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            if smile_type:
                type_color = (0, 255, 0) if smile_type == "Genuine" else (0, 165, 255) if smile_type == "Fake" else (0, 255, 255)
                cv2.putText(frame, f'Type: {smile_type}', (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, type_color, 2)
            
            cv2.imshow('Smile Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    final_score = sum(smile_scores) / len(smile_scores) if smile_scores else 0.0
    print(f"\nSession ended. Average smile score: {final_score:.3f}")


def detect_smile_from_image(image_path):
    """
    Detect smile from a static image using MediaPipe landmarks
    
    Args:
        image_path: Path to the image file
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print("No faces detected in the image")
            return
        
        for face_landmarks in results.multi_face_landmarks:
            # Calculate smile score from landmarks (no baseline for static images)
            smile_score, lip_coords, mouth_width = calculate_smile_score_from_landmarks(
                face_landmarks, w, h, None
            )
            
            # Get face bounding box
            x_coords = [lm.x * w for lm in face_landmarks.landmark]
            y_coords = [lm.y * h for lm in face_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Draw face bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image, 'Face', (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw ALL lip coordinates
            if lip_coords:
                left_corner = lip_coords['left_corner']
                right_corner = lip_coords['right_corner']
                top_center = lip_coords['top_center']
                bottom_center = lip_coords['bottom_center']
                
                # Draw lip corners (green circles)
                cv2.circle(image, left_corner, 5, (0, 255, 0), -1)
                cv2.circle(image, right_corner, 5, (0, 255, 0), -1)
                
                # Draw line connecting corners
                cv2.line(image, left_corner, right_corner, (0, 255, 0), 2)
                
                # Draw top and bottom lip centers (blue circles)
                cv2.circle(image, top_center, 4, (255, 0, 0), -1)
                cv2.circle(image, bottom_center, 4, (255, 0, 0), -1)
                
                # Draw additional key lip points (yellow)
                cv2.circle(image, lip_coords['top_lip_left'], 3, (0, 255, 255), -1)
                cv2.circle(image, lip_coords['top_lip_right'], 3, (0, 255, 255), -1)
                cv2.circle(image, lip_coords['bottom_lip_left'], 3, (0, 255, 255), -1)
                cv2.circle(image, lip_coords['bottom_lip_right'], 3, (0, 255, 255), -1)
                
                # Draw ALL mouth landmarks (small cyan dots)
                if 'all_landmarks' in lip_coords:
                    for landmark_name, coord in lip_coords['all_landmarks'].items():
                        cv2.circle(image, coord, 2, (255, 255, 0), -1)
                
                # Display key coordinates
                coord_text = f'L:{left_corner[0]},{left_corner[1]} R:{right_corner[0]},{right_corner[1]}'
                cv2.putText(image, coord_text, (x_min, y_max + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Print ALL coordinates to console
                print(f"Key Lip coordinates:")
                print(f"  Left corner: {left_corner}")
                print(f"  Right corner: {right_corner}")
                print(f"  Top center: {top_center}")
                print(f"  Bottom center: {bottom_center}")
                print(f"  Top lip left: {lip_coords['top_lip_left']}")
                print(f"  Top lip right: {lip_coords['top_lip_right']}")
                print(f"  Bottom lip left: {lip_coords['bottom_lip_left']}")
                print(f"  Bottom lip right: {lip_coords['bottom_lip_right']}")
                
                if 'all_landmarks' in lip_coords:
                    print(f"\nAll Lip Landmarks ({len(lip_coords['all_landmarks'])} points):")
                    for landmark_name, coord in lip_coords['all_landmarks'].items():
                        print(f"  {landmark_name}: {coord}")
            
            # Binary classification: No Smile or Smile
            # Threshold based on width increase (0.5 = 15% increase threshold)
            if smile_score >= 0.5:
                smile_text = f'Smile (Score: {smile_score:.3f})'
                smile_color = (0, 255, 0)  # Green
            else:
                smile_text = f'No Smile (Score: {smile_score:.3f})'
                smile_color = (0, 0, 255)  # Red
            
            cv2.putText(image, smile_text, (x_min, y_min - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, smile_color, 2)
            
            # Display mouth width
            cv2.putText(image, f'Mouth Width: {mouth_width:.1f}px', (x_min, y_max + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print(f"Face detected - Smile Score: {smile_score:.3f} - {'Smile' if smile_score >= 0.5 else 'No Smile'}")
            print(f"Mouth Width: {mouth_width:.1f}px")
    
    # Display result
    cv2.imshow('Smile Detection Result', image)
    print("Press any key to close the window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If image path provided, detect from image
        image_path = sys.argv[1]
        detect_smile_from_image(image_path)
    else:
        # Otherwise, use webcam
        detect_smile()