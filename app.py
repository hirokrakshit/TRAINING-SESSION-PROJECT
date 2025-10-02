import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(page_title="AI Fitness Coach", layout="wide")
st.title("🏋️ AI Fitness Coach – Pose Detection")
st.markdown("Real-time posture detection and rep counting using MediaPipe + OpenCV")

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

def process_frame(image, exercise):
    """Process a single frame for pose detection"""
    # Convert to RGB
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        
        # Convert back to BGR for OpenCV drawing
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                if exercise == "Push-ups":
                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    
                    # Rep counting logic
                    if angle > 160:
                        st.session_state.stage = "up"
                    if angle < 90 and st.session_state.stage == "up":
                        st.session_state.stage = "down"
                        st.session_state.counter += 1
                        
                elif exercise == "Squats":
                    # Get coordinates
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = calculate_angle(left_hip, left_knee, left_ankle)
                    
                    # Rep counting logic
                    if angle > 170:
                        st.session_state.stage = "up"
                    if angle < 90 and st.session_state.stage == "up":
                        st.session_state.stage = "down"
                        st.session_state.counter += 1
                
                # Draw stats box
                cv2.rectangle(image_bgr, (0,0), (225,73), (245,117,16), -1)
                
                # Display REPS
                cv2.putText(image_bgr, 'REPS', (15,20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, str(st.session_state.counter),
                          (10,60),
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Display STAGE
                cv2.putText(image_bgr, 'STAGE', (65,20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, st.session_state.stage if st.session_state.stage else "-",
                          (60,60),
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
            except Exception as e:
                st.error(f"Error processing landmarks: {e}")
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image_bgr, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        return image_bgr

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
        st.image(processed_image, channels="BGR", use_container_width=True)

else:  # Webcam Snapshot
    st.info("📸 Click the button below to take a snapshot from your webcam")
    
    camera_photo = st.camera_input("Take a picture")
    
    if camera_photo is not None:
        # Read image
        image = Image.open(camera_photo)
        
        # Process frame
        processed_image = process_frame(image, exercise)
        
        # Display
        st.image(processed_image, channels="BGR", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit, MediaPipe, and OpenCV</p>
    <p>💡 Tip: Make sure your full body is visible in the frame for best results</p>
</div>
""", unsafe_allow_html=True)
