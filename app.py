import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import json

# Page configuration
st.set_page_config(page_title="AI Fitness Coach", layout="wide")
st.title("🏋️ AI Fitness Coach – Pose Detection")
st.markdown("Real-time posture detection and rep counting using Pose Estimation")

# Initialize session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'stage' not in st.session_state:
    st.session_state.stage = None

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_poses_movenet(image):
    """Detect poses using MoveNet model via TensorFlow Hub"""
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        
        # Load MoveNet model
        @st.cache_resource
        def load_model():
            model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
            return model
        
        model = load_model()
        movenet = model.signatures['serving_default']
        
        # Preprocess image
        img = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 192, 192)
        img = tf.cast(img, dtype=tf.int32)
        
        # Run inference
        outputs = movenet(img)
        keypoints = outputs['output_0'].numpy()[0][0]
        
        return keypoints
    except Exception as e:
        st.error(f"Error loading TensorFlow model: {e}")
        return None

def detect_poses_opencv(image):
    """Detect poses using OpenCV DNN with OpenPose model"""
    try:
        # This is a simplified version - you would need the OpenPose model files
        # For deployment, we'll use a simpler approach
        st.warning("OpenCV DNN pose detection requires model files. Using alternative method.")
        return None
    except Exception as e:
        st.error(f"Error with OpenCV detection: {e}")
        return None

def draw_keypoints(image, keypoints, confidence_threshold=0.3):
    """Draw detected keypoints on image"""
    h, w = image.shape[:2]
    
    # MoveNet keypoint indices
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    connections = [
        (5, 7), (7, 9),   # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 6),           # Shoulders
        (5, 11), (6, 12), # Torso
        (11, 12),         # Hips
        (11, 13), (13, 15), # Left leg
        (12, 14), (14, 16)  # Right leg
    ]
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if keypoints[start_idx][2] > confidence_threshold and keypoints[end_idx][2] > confidence_threshold:
            start_point = (int(keypoints[start_idx][1] * w), int(keypoints[start_idx][0] * h))
            end_point = (int(keypoints[end_idx][1] * w), int(keypoints[end_idx][0] * h))
            cv2.line(image, start_point, end_point, (245, 66, 230), 2)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp[2] > confidence_threshold:
            x, y = int(kp[1] * w), int(kp[0] * h)
            cv2.circle(image, (x, y), 4, (245, 117, 66), -1)
    
    return image

def process_frame(image, exercise):
    """Process a single frame for pose detection"""
    # Convert to RGB array
    image_array = np.array(image)
    if len(image_array.shape) == 2:  # Grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Detect poses
    keypoints = detect_poses_movenet(image_array)
    
    if keypoints is not None:
        h, w = image_array.shape[:2]
        
        try:
            if exercise == "Push-ups":
                # Keypoint indices: 5=left_shoulder, 7=left_elbow, 9=left_wrist
                left_shoulder = [keypoints[5][1] * w, keypoints[5][0] * h]
                left_elbow = [keypoints[7][1] * w, keypoints[7][0] * h]
                left_wrist = [keypoints[9][1] * w, keypoints[9][0] * h]
                
                if keypoints[5][2] > 0.3 and keypoints[7][2] > 0.3 and keypoints[9][2] > 0.3:
                    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    
                    # Rep counting logic
                    if angle > 160:
                        st.session_state.stage = "up"
                    if angle < 90 and st.session_state.stage == "up":
                        st.session_state.stage = "down"
                        st.session_state.counter += 1
                        
            elif exercise == "Squats":
                # Keypoint indices: 11=left_hip, 13=left_knee, 15=left_ankle
                left_hip = [keypoints[11][1] * w, keypoints[11][0] * h]
                left_knee = [keypoints[13][1] * w, keypoints[13][0] * h]
                left_ankle = [keypoints[15][1] * w, keypoints[15][0] * h]
                
                if keypoints[11][2] > 0.3 and keypoints[13][2] > 0.3 and keypoints[15][2] > 0.3:
                    angle = calculate_angle(left_hip, left_knee, left_ankle)
                    
                    # Rep counting logic
                    if angle > 170:
                        st.session_state.stage = "up"
                    if angle < 90 and st.session_state.stage == "up":
                        st.session_state.stage = "down"
                        st.session_state.counter += 1
            
        except Exception as e:
            st.warning(f"Could not calculate angles: {e}")
        
        # Draw keypoints
        image_array = draw_keypoints(image_array, keypoints)
    
    # Draw stats box
    cv2.rectangle(image_array, (0, 0), (225, 73), (245, 117, 16), -1)
    
    # Display REPS
    cv2.putText(image_array, 'REPS', (15, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image_array, str(st.session_state.counter),
              (10, 60),
              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display STAGE
    cv2.putText(image_array, 'STAGE', (65, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image_array, st.session_state.stage if st.session_state.stage else "-",
              (60, 60),
              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image_array

# Sidebar controls
st.sidebar.header("Controls")
exercise = st.sidebar.selectbox("Choose Exercise", ["Push-ups", "Squats"])

if st.sidebar.button("Reset Counter"):
    st.session_state.counter = 0
    st.session_state.stage = None
    st.rerun()

# Display current stats
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Reps", st.session_state.counter)
with col2:
    st.metric("Stage", st.session_state.stage if st.session_state.stage else "-")

# Instructions
with st.sidebar.expander("📋 Instructions"):
    if exercise == "Push-ups":
        st.write("""
        **Push-up Form:**
        - Start in plank position
        - Lower body until arms at 90°
        - Push back up to starting position
        - Keep body straight throughout
        """)
    else:
        st.write("""
        **Squat Form:**
        - Stand with feet shoulder-width apart
        - Lower hips until knees at 90°
        - Push back up to standing position
        - Keep chest up and back straight
        """)

# Main content area
st.markdown("---")

# Camera input method selection
camera_method = st.radio(
    "Select Camera Input Method:",
    ["Upload Image", "Webcam Snapshot"],
    horizontal=True
)

if camera_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        # Process frame
        processed_image = process_frame(image, exercise)
        
        # Display
        st.image(processed_image, channels="RGB", use_container_width=True)

else:  # Webcam Snapshot
    st.info("📸 Click the button below to take a snapshot from your webcam")
    
    camera_photo = st.camera_input("Take a picture")
    
    if camera_photo is not None:
        # Read image
        image = Image.open(camera_photo)
        
        # Process frame
        processed_image = process_frame(image, exercise)
        
        # Display
        st.image(processed_image, channels="RGB", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit, TensorFlow, and OpenCV</p>
    <p>💡 Tip: Make sure your full body is visible in the frame for best results</p>
</div>
""", unsafe_allow_html=True)
