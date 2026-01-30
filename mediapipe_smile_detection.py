"""
Smile Detection using MediaPipe Face Mesh
More accurate approach using facial landmarks
"""

import cv2
import mediapipe as mp
import numpy as np
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


def calculate_smile_ratio(face_landmarks, image_width, image_height):
    """
    Calculate smile ratio based on mouth landmarks
    
    Args:
        face_landmarks: MediaPipe face landmarks
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        smile_ratio: Ratio indicating smile intensity (0-1)
    """
    # MediaPipe face mesh landmark indices for mouth
    # Left mouth corner: 61
    # Right mouth corner: 291
    # Top lip center: 13
    # Bottom lip center: 14
    
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    
    # Calculate mouth width
    mouth_width = np.sqrt(
        (right_mouth.x - left_mouth.x)**2 * image_width**2 +
        (right_mouth.y - left_mouth.y)**2 * image_height**2
    )
    
    # Calculate mouth height (vertical distance between lips)
    mouth_height = np.sqrt(
        (bottom_lip.x - top_lip.x)**2 * image_width**2 +
        (bottom_lip.y - top_lip.y)**2 * image_height**2
    )
    
    # Smile ratio: wider mouth with less height = bigger smile
    if mouth_height > 0:
        smile_ratio = mouth_width / (mouth_height + 1)  # Add 1 to avoid division by zero
    else:
        smile_ratio = 0
    
    return smile_ratio, (left_mouth, right_mouth, top_lip, bottom_lip)


def detect_smile_mediapipe():
    """
    Main function to detect smiles using MediaPipe Face Mesh
    """
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
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
    
    print("MediaPipe Smile Detection Started!")
    print("Press 'q' to quit")
    
    smile_threshold = 15.0  # Adjust this value to change sensitivity
    smile_count = 0
    frame_count = 0
    
    # Eye aperture tracking for genuine/fake smile detection
    baseline_eye_aperture = None  # Baseline eye aperture (neutral expression)
    baseline_eye_apertures = deque(maxlen=30)  # Track baseline eye aperture over time
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # Flip image horizontally for mirror effect
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            h, w, _ = image.shape
            
            smile_detected = False
            current_eye_aperture = 0.0
            eye_coords = None
            smile_type = ""  # "Genuine" or "Fake"
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate smile ratio
                    smile_ratio, mouth_points = calculate_smile_ratio(
                        face_landmarks, w, h
                    )
                    
                    # Calculate eye aperture
                    avg_aperture, left_aperture, right_aperture, eye_coords = calculate_eye_aperture(
                        face_landmarks, w, h
                    )
                    current_eye_aperture = avg_aperture
                    
                    # Update baseline eye aperture when not smiling
                    if smile_ratio <= smile_threshold:
                        baseline_eye_apertures.append(current_eye_aperture)
                        if len(baseline_eye_apertures) > 5:  # After 5 frames, establish baseline
                            baseline_eye_aperture = sum(baseline_eye_apertures) / len(baseline_eye_apertures)
                    elif baseline_eye_aperture is None and len(baseline_eye_apertures) > 0:
                        # If no baseline yet, use current aperture as initial baseline
                        baseline_eye_aperture = sum(baseline_eye_apertures) / len(baseline_eye_apertures) if baseline_eye_apertures else current_eye_aperture
                    
                    # Draw face mesh (optional - comment out if you don't want to see all landmarks)
                    # mp_drawing.draw_landmarks(
                    #     image,
                    #     face_landmarks,
                    #     mp_face_mesh.FACEMESH_CONTOURS,
                    #     None,
                    #     mp_drawing_styles.get_default_face_mesh_contours_style()
                    # )
                    
                    # Draw mouth landmarks
                    left_mouth, right_mouth, top_lip, bottom_lip = mouth_points
                    
                    # Convert normalized coordinates to pixel coordinates
                    left_mouth_px = (int(left_mouth.x * w), int(left_mouth.y * h))
                    right_mouth_px = (int(right_mouth.x * w), int(right_mouth.y * h))
                    top_lip_px = (int(top_lip.x * w), int(top_lip.y * h))
                    bottom_lip_px = (int(bottom_lip.x * w), int(bottom_lip.y * h))
                    
                    # Draw mouth corners
                    cv2.circle(image, left_mouth_px, 5, (0, 255, 0), -1)
                    cv2.circle(image, right_mouth_px, 5, (0, 255, 0), -1)
                    cv2.line(image, left_mouth_px, right_mouth_px, (0, 255, 0), 2)
                    
                    # Determine if smiling
                    if smile_ratio > smile_threshold:
                        smile_detected = True
                        # Check if it's genuine or fake smile
                        if baseline_eye_aperture and baseline_eye_aperture > 0:
                            # Calculate eye aperture change (percentage decrease)
                            eye_aperture_decrease = ((baseline_eye_aperture - current_eye_aperture) / baseline_eye_aperture) * 100
                            
                            # Genuine smile: eyes become smaller (aperture decreases by at least 5%)
                            # Fake smile: eyes don't become smaller (aperture stays same or increases)
                            if eye_aperture_decrease >= 5.0:  # 5% decrease threshold
                                status_text = f'Genuine Smile! :) (Ratio: {smile_ratio:.1f})'
                                color = (0, 255, 0)  # Green
                                smile_type = "Genuine"
                            else:
                                status_text = f'Fake Smile (Ratio: {smile_ratio:.1f})'
                                color = (0, 165, 255)  # Orange
                                smile_type = "Fake"
                        else:
                            # Baseline not established yet
                            status_text = f'Smiling! :) (Analyzing...) (Ratio: {smile_ratio:.1f})'
                            color = (0, 255, 255)  # Yellow
                            smile_type = "Unknown"
                    else:
                        status_text = f'Not Smiling (Ratio: {smile_ratio:.1f})'
                        color = (0, 0, 255)  # Red
                        smile_type = ""
                    
                    # Display status
                    cv2.putText(image, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw eye landmarks for visualization
                    if eye_coords:
                        # Draw left eye (top and bottom)
                        cv2.circle(image, eye_coords['left_eye_top'], 4, (255, 255, 0), -1)
                        cv2.circle(image, eye_coords['left_eye_bottom'], 4, (255, 255, 0), -1)
                        cv2.line(image, eye_coords['left_eye_top'], eye_coords['left_eye_bottom'], (255, 255, 0), 2)
                        
                        # Draw right eye (top and bottom)
                        cv2.circle(image, eye_coords['right_eye_top'], 4, (255, 255, 0), -1)
                        cv2.circle(image, eye_coords['right_eye_bottom'], 4, (255, 255, 0), -1)
                        cv2.line(image, eye_coords['right_eye_top'], eye_coords['right_eye_bottom'], (255, 255, 0), 2)
                    
                    # Display eye aperture information
                    if baseline_eye_aperture and baseline_eye_aperture > 0:
                        eye_decrease = ((baseline_eye_aperture - current_eye_aperture) / baseline_eye_aperture) * 100
                        eye_text = f'Eye Aperture: {current_eye_aperture:.1f}px (Baseline: {baseline_eye_aperture:.1f}px, {eye_decrease:+.1f}%)'
                        eye_color = (0, 255, 0) if eye_decrease >= 5.0 else (0, 165, 255) if smile_detected else (255, 255, 255)
                        cv2.putText(image, eye_text, (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)
                    
                    if smile_type:
                        type_color = (0, 255, 0) if smile_type == "Genuine" else (0, 165, 255) if smile_type == "Fake" else (0, 255, 255)
                        cv2.putText(image, f'Type: {smile_type}', (10, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_color, 2)
            
            # Update statistics
            frame_count += 1
            if smile_detected:
                smile_count += 1
            
            smile_percentage = (smile_count / frame_count * 100) if frame_count > 0 else 0
            
            # Display statistics
            cv2.putText(image, f'Smile Rate: {smile_percentage:.1f}%', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f'Threshold: {smile_threshold}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(image, "Press 'q' to quit, '+' to increase threshold, '-' to decrease",
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('MediaPipe Smile Detection', image)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                smile_threshold += 0.5
                print(f"Threshold increased to: {smile_threshold}")
            elif key == ord('-') or key == ord('_'):
                smile_threshold = max(0.5, smile_threshold - 0.5)
                print(f"Threshold decreased to: {smile_threshold}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Smile rate: {smile_percentage:.1f}%")


if __name__ == "__main__":
    detect_smile_mediapipe()

