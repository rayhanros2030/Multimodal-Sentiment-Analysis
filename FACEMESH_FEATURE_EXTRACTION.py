"""
Example Facemesh Feature Extraction for Emotion Recognition

This script shows how to extract emotion-relevant features from MediaPipe Facemesh 468 landmarks.
"""

import numpy as np
import mediapipe as mp

def extract_emotion_features_from_landmarks(landmarks, face_width=None):
    """
    Extract emotion-relevant features from Facemesh 468 landmarks.
    
    Args:
        landmarks: Array of shape [468, 3] with (x, y, z) coordinates
        face_width: Optional face width for normalization
    
    Returns:
        features: Array of ~65 emotion-relevant features
    """
    
    # Normalize face size if face_width provided
    if face_width is None:
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])  # Distance between face edges
    
    normalized_landmarks = landmarks / face_width
    
    features = []
    
    # ========== MOUTH FEATURES (Most Important for Emotion) ==========
    
    # Mouth width (landmarks 61, 291)
    mouth_width = np.linalg.norm(normalized_landmarks[61] - normalized_landmarks[291])
    features.append(mouth_width)
    
    # Mouth height (landmarks 13, 14)
    mouth_height = np.linalg.norm(normalized_landmarks[13] - normalized_landmarks[14])
    features.append(mouth_height)
    
    # Mouth corner positions (for smile/frown detection)
    left_corner_y = normalized_landmarks[61, 1]  # Y position of left corner
    right_corner_y = normalized_landmarks[291, 1]  # Y position of right corner
    features.extend([left_corner_y, right_corner_y])
    
    # Mouth corner angle (smile vs frown)
    mouth_center_y = (normalized_landmarks[13, 1] + normalized_landmarks[14, 1]) / 2
    corner_angle = np.arctan2(
        (left_corner_y + right_corner_y) / 2 - mouth_center_y,
        mouth_width / 2
    )
    features.append(corner_angle)
    
    # Upper lip features
    upper_lip_height = np.mean([
        normalized_landmarks[61, 1], normalized_landmarks[78, 1],
        normalized_landmarks[95, 1], normalized_landmarks[88, 1],
        normalized_landmarks[178, 1], normalized_landmarks[87, 1],
        normalized_landmarks[14, 1], normalized_landmarks[317, 1],
        normalized_landmarks[402, 1], normalized_landmarks[318, 1],
        normalized_landmarks[324, 1]
    ])
    features.append(upper_lip_height)
    
    # Lower lip features
    lower_lip_height = np.mean([
        normalized_landmarks[78, 1], normalized_landmarks[95, 1],
        normalized_landmarks[88, 1], normalized_landmarks[178, 1],
        normalized_landmarks[87, 1], normalized_landmarks[14, 1],
        normalized_landmarks[317, 1], normalized_landmarks[402, 1],
        normalized_landmarks[318, 1], normalized_landmarks[324, 1]
    ])
    features.append(lower_lip_height)
    
    # Lip thickness
    lip_thickness = abs(upper_lip_height - lower_lip_height)
    features.append(lip_thickness)
    
    # ========== EYE FEATURES ==========
    
    # Left eye width (landmarks 33, 133)
    left_eye_width = np.linalg.norm(normalized_landmarks[33] - normalized_landmarks[133])
    features.append(left_eye_width)
    
    # Right eye width (landmarks 362, 263)
    right_eye_width = np.linalg.norm(normalized_landmarks[362] - normalized_landmarks[263])
    features.append(right_eye_width)
    
    # Left eye height (vertical distance)
    left_eye_top = normalized_landmarks[159, 1]
    left_eye_bottom = normalized_landmarks[145, 1]
    left_eye_height = abs(left_eye_top - left_eye_bottom)
    features.append(left_eye_height)
    
    # Right eye height
    right_eye_top = normalized_landmarks[386, 1]
    right_eye_bottom = normalized_landmarks[374, 1]
    right_eye_height = abs(right_eye_top - right_eye_bottom)
    features.append(right_eye_height)
    
    # Eye opening ratio (for surprise/sleep)
    left_eye_opening = left_eye_height / left_eye_width if left_eye_width > 0 else 0
    right_eye_opening = right_eye_height / right_eye_width if right_eye_width > 0 else 0
    features.extend([left_eye_opening, right_eye_opening])
    
    # Inter-eye distance (landmarks 33, 263)
    inter_eye_distance = np.linalg.norm(normalized_landmarks[33] - normalized_landmarks[263])
    features.append(inter_eye_distance)
    
    # ========== EYEBROW FEATURES ==========
    
    # Left eyebrow height
    left_eyebrow_inner = normalized_landmarks[21, 1]
    left_eyebrow_outer = normalized_landmarks[107, 1]
    left_eyebrow_center = normalized_landmarks[55, 1]
    left_eyebrow_height = np.mean([left_eyebrow_inner, left_eyebrow_center, left_eyebrow_outer])
    features.append(left_eyebrow_height)
    
    # Right eyebrow height
    right_eyebrow_inner = normalized_landmarks[251, 1]
    right_eyebrow_outer = normalized_landmarks[336, 1]
    right_eyebrow_center = normalized_landmarks[285, 1]
    right_eyebrow_height = np.mean([right_eyebrow_inner, right_eyebrow_center, right_eyebrow_outer])
    features.append(right_eyebrow_height)
    
    # Eyebrow angle (for frown/surprise)
    left_eyebrow_angle = np.arctan2(
        left_eyebrow_outer - left_eyebrow_inner,
        np.linalg.norm(normalized_landmarks[107] - normalized_landmarks[21])
    )
    right_eyebrow_angle = np.arctan2(
        right_eyebrow_outer - right_eyebrow_inner,
        np.linalg.norm(normalized_landmarks[336] - normalized_landmarks[251])
    )
    features.extend([left_eyebrow_angle, right_eyebrow_angle])
    
    # Eyebrow to eye distance (for surprise)
    left_eyebrow_eye_distance = abs(left_eyebrow_center - left_eye_top)
    right_eyebrow_eye_distance = abs(right_eyebrow_center - right_eye_top)
    features.extend([left_eyebrow_eye_distance, right_eyebrow_eye_distance])
    
    # ========== SYMMETRY FEATURES ==========
    
    # Facial symmetry (left vs right features)
    eye_symmetry = abs(left_eye_width - right_eye_width) / max(left_eye_width, right_eye_width) if max(left_eye_width, right_eye_width) > 0 else 0
    eyebrow_symmetry = abs(left_eyebrow_height - right_eyebrow_height)
    features.extend([eye_symmetry, eyebrow_symmetry])
    
    # Mouth symmetry
    mouth_left_half = np.linalg.norm(normalized_landmarks[61] - normalized_landmarks[13])
    mouth_right_half = np.linalg.norm(normalized_landmarks[291] - normalized_landmarks[14])
    mouth_symmetry = abs(mouth_left_half - mouth_right_half) / max(mouth_left_half, mouth_right_half) if max(mouth_left_half, mouth_right_half) > 0 else 0
    features.append(mouth_symmetry)
    
    # ========== RATIO FEATURES ==========
    
    # Mouth to face width ratio
    mouth_face_ratio = mouth_width / face_width if face_width > 0 else 0
    features.append(mouth_face_ratio)
    
    # Eye to face width ratio
    eye_face_ratio = inter_eye_distance / face_width if face_width > 0 else 0
    features.append(eye_face_ratio)
    
    # ========== POSE FEATURES ==========
    
    # Head rotation (approximate from key points)
    nose_tip = normalized_landmarks[1]
    chin = normalized_landmarks[152]
    head_tilt = np.arctan2(chin[0] - nose_tip[0], chin[1] - nose_tip[1])
    features.append(head_tilt)
    
    # Face orientation (yaw approximation)
    left_face_edge = normalized_landmarks[0, 0]
    right_face_edge = normalized_landmarks[16, 0]
    face_center_x = (left_face_edge + right_face_edge) / 2
    nose_x = normalized_landmarks[1, 0]
    head_yaw = (nose_x - face_center_x) / face_width if face_width > 0 else 0
    features.append(head_yaw)
    
    return np.array(features, dtype=np.float32)


def process_video_frames_facemesh(video_path, extract_features_fn):
    """
    Process video frames and extract features from Facemesh.
    
    Args:
        video_path: Path to video file
        extract_features_fn: Function to extract features from landmarks
    
    Returns:
        frame_features: Array of shape [num_frames, feature_dim]
    """
    import cv2
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(str(video_path))
    frame_features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with Facemesh
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Extract landmarks
            landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.multi_face_landmarks[0].landmark
            ])
            
            # Extract features
            features = extract_features_fn(landmarks)
            frame_features.append(features)
        else:
            # No face detected, use zeros
            frame_features.append(np.zeros(65, dtype=np.float32))
    
    cap.release()
    face_mesh.close()
    
    return np.array(frame_features)  # Shape: [num_frames, feature_dim]


def temporal_average_features(frame_features):
    """
    Average features across frames (current approach).
    
    Args:
        frame_features: Array of shape [num_frames, feature_dim]
    
    Returns:
        averaged_features: Array of shape [feature_dim]
    """
    return np.mean(frame_features, axis=0)


def temporal_pool_features(frame_features):
    """
    Pool features across frames (mean + max + std).
    
    Args:
        frame_features: Array of shape [num_frames, feature_dim]
    
    Returns:
        pooled_features: Array of shape [feature_dim * 3]
    """
    mean_feat = np.mean(frame_features, axis=0)
    max_feat = np.max(frame_features, axis=0)
    std_feat = np.std(frame_features, axis=0)
    
    return np.concatenate([mean_feat, max_feat, std_feat])


# Example usage:
if __name__ == "__main__":
    # For a single frame (landmarks already extracted):
    landmarks = np.random.rand(468, 3)  # Example: 468 landmarks × 3 coords
    features = extract_emotion_features_from_landmarks(landmarks)
    print(f"Extracted {len(features)} emotion features")
    
    # For a video (process frame by frame):
    # frame_features = process_video_frames_facemesh("video.mp4", extract_emotion_features_from_landmarks)
    # averaged = temporal_average_features(frame_features)  # [65]
    # pooled = temporal_pool_features(frame_features)  # [195]

