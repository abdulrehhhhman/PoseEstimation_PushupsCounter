import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from PIL import Image

# --- Pose Estimation Keypoint Definitions ---
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# --- Enhanced Helper Functions ---
def calculate_angle(p1, p2, p3):
    """
    Calculates the angle (in degrees) between three points using vectors.
    p2 is the vertex of the angle.
    Returns angle between 0-180 degrees.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle

def calculate_body_alignment_angle(shoulder, hip, ankle):
    """
    Calculate the angle of body alignment (useful for side views)
    """
    return calculate_angle(shoulder, hip, ankle)

def calculate_vertical_displacement(keypoints, reference_point_idx, 
                                   current_point_idx):
    """
    Calculate vertical displacement between frames
    """
    if len(keypoints) > max(reference_point_idx, current_point_idx):
        ref_point = keypoints[reference_point_idx]
        curr_point = keypoints[current_point_idx]
        return abs(ref_point[1] - curr_point[1])  # Y-axis displacement
    return 0

def detect_side_view_orientation(keypoints):
    """
    Detect if the person is in side view based on shoulder visibility
    """
    left_shoulder = keypoints[LEFT_SHOULDER]
    right_shoulder = keypoints[RIGHT_SHOULDER]
    
    # Calculate horizontal distance between shoulders
    shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
    
    # In side view, shoulders overlap, so distance is small
    # Threshold based on typical shoulder width in pixels
    is_side_view = shoulder_distance < 50  # Adjust based on video resolution
    
    return is_side_view

def check_keypoint_quality(keypoints, required_points, 
                           confidence_threshold=0.3):
    """
    Check if required keypoints are detected with sufficient quality
    """
    for idx in required_points:
        if len(keypoints) <= idx:
            return False
        point = keypoints[idx]
        if np.allclose(point, [0, 0]) or np.any(np.isnan(point)) or np.any(np.isinf(point)):
            return False
    return True

# Page configuration
st.set_page_config(
    page_title="💪 AI Push-up Counter",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .pushup-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .counter-display {
        font-size: 4rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pushup_history' not in st.session_state:
    st.session_state.pushup_history = []

if 'current_pushup_count' not in st.session_state:
    st.session_state.current_pushup_count = 0

if 'uploaded_model_temp_path' not in st.session_state:
    st.session_state.uploaded_model_temp_path = None

class EnhancedPushupCounter:
    def __init__(self, model_path=None):
        """Initialize the enhanced push-up counter with YOLO pose model"""
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                st.error(f"Error loading model from {model_path}: {str(e)}")
        
        # Push-up counting variables
        self.pushup_count = 0
        self.pushup_stage = None  # 'up' or 'down'
        
        # Adaptive angle thresholds
        self.PUSHUP_THRESHOLD_DOWN = 90
        self.PUSHUP_THRESHOLD_UP = 140
        
        # Side view specific thresholds
        self.SIDE_VIEW_THRESHOLD_DOWN = 100
        self.SIDE_VIEW_THRESHOLD_UP = 160
        
        # Vertical displacement thresholds
        self.MIN_VERTICAL_DISPLACEMENT = 20  # pixels
        
        # Smoothing
        self.angle_history = []
        self.angle_window_size = 5
        
        # State tracking
        self.min_frames_between_reps = 15
        self.frames_since_last_rep = 0
        self.is_side_view = False
        
        # Body alignment tracking
        self.body_alignment_history = []
        
        # Hip tracking for side view
        self.hip_position_history = []
        self.hip_window_size = 5

    def detect_pushup_pose(self, frame, confidence_threshold=0.5):
        """
        Detect human poses in the frame using the YOLO model.
        """
        if self.model is None:
            return [], frame.copy() 
        
        try:
            results = self.model(frame)
            
            keypoints_data_list = []
            annotated_frame = frame.copy()

            if results and len(results) > 0:
                annotated_frame = results[0].plot()

                for r in results:
                    if r.keypoints and r.keypoints.xy.numel() > 0:
                        
                        if len(r.boxes) > 0: 
                            person_confidence = float(r.boxes[0].conf[0])
                        else:
                            person_confidence = 0

                        if person_confidence > confidence_threshold:
                            keypoints_data_list.append({
                                'keypoints': r.keypoints.xy[0].cpu().numpy(),
                                'confidence': person_confidence
                            })
            
            return keypoints_data_list, annotated_frame
            
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return [], frame.copy()

    def count_pushup_enhanced(self, keypoints_data_list):
        """
        Enhanced push-up counting with better side-view support
        """
        self.frames_since_last_rep += 1
        
        if not keypoints_data_list:
            return self.pushup_count

        keypoints = keypoints_data_list[0]['keypoints']
        
        # Detect viewing angle
        self.is_side_view = detect_side_view_orientation(keypoints)
        
        # Choose appropriate keypoints based on view
        if self.is_side_view:
            # For side view, prioritize visible side
            required_kpts = []
            if not np.allclose(keypoints[LEFT_SHOULDER], [0, 0]):
                required_kpts = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP]
                primary_shoulder = LEFT_SHOULDER
                primary_elbow = LEFT_ELBOW
                primary_wrist = LEFT_WRIST
                primary_hip = LEFT_HIP
            else:
                required_kpts = [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP]
                primary_shoulder = RIGHT_SHOULDER
                primary_elbow = RIGHT_ELBOW
                primary_wrist = RIGHT_WRIST
                primary_hip = RIGHT_HIP
        else:
            # Front/back view - use both sides
            required_kpts = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, 
                              RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]

        if not check_keypoint_quality(keypoints, required_kpts):
            return self.pushup_count
        
        try:
            if self.is_side_view:
                # Side view analysis
                elbow_angle = calculate_angle(
                    keypoints[primary_shoulder], 
                    keypoints[primary_elbow], 
                    keypoints[primary_wrist]
                )
                
                # Track hip vertical position
                hip_y = keypoints[primary_hip][1]
                self.hip_position_history.append(hip_y)
                if len(self.hip_position_history) > self.hip_window_size:
                    self.hip_position_history.pop(0)
                
                # Calculate hip displacement
                if len(self.hip_position_history) >= 2:
                    hip_displacement = max(self.hip_position_history) - min(self.hip_position_history)
                else:
                    hip_displacement = 0
                
                # Body alignment check
                if len(keypoints) > max(primary_shoulder, primary_hip, LEFT_KNEE):
                    body_angle = calculate_body_alignment_angle(
                        keypoints[primary_shoulder],
                        keypoints[primary_hip],
                        keypoints[LEFT_KNEE] if LEFT_KNEE < len(keypoints) else keypoints[primary_hip]
                    )
                    self.body_alignment_history.append(body_angle)
                    if len(self.body_alignment_history) > 5:
                        self.body_alignment_history.pop(0)
                
                # Use side-view specific thresholds
                threshold_down = self.SIDE_VIEW_THRESHOLD_DOWN
                threshold_up = self.SIDE_VIEW_THRESHOLD_UP
                
                # Combine angle and displacement for better detection
                use_displacement = hip_displacement > self.MIN_VERTICAL_DISPLACEMENT
                
            else:
                # Front/back view analysis (original logic)
                right_elbow_angle = calculate_angle(
                    keypoints[RIGHT_SHOULDER], 
                    keypoints[RIGHT_ELBOW], 
                    keypoints[RIGHT_WRIST]
                )
                left_elbow_angle = calculate_angle(
                    keypoints[LEFT_SHOULDER], 
                    keypoints[LEFT_ELBOW], 
                    keypoints[LEFT_WRIST]
                )
                elbow_angle = (right_elbow_angle + left_elbow_angle) / 2
                threshold_down = self.PUSHUP_THRESHOLD_DOWN
                threshold_up = self.PUSHUP_THRESHOLD_UP
                use_displacement = False
                hip_displacement = 0
            
            # Apply smoothing
            self.angle_history.append(elbow_angle)
            if len(self.angle_history) > self.angle_window_size:
                self.angle_history.pop(0)
            
            smooth_angle = np.mean(self.angle_history)
            
            # Enhanced debugging
            st.sidebar.info(f"View Type: {'Side' if self.is_side_view else 'Front/Back'}")
            st.sidebar.info(f"Elbow Angle: {elbow_angle:.1f}°")
            st.sidebar.info(f"Smooth Angle: {smooth_angle:.1f}°")
            if self.is_side_view:
                st.sidebar.info(f"Hip Displacement: {hip_displacement:.1f} pixels")
            st.sidebar.info(f"Current Stage: {self.pushup_stage}")
            st.sidebar.info(f"Frames since last rep: {self.frames_since_last_rep}")
            
            # State machine with enhanced logic
            if (smooth_angle < threshold_down or (self.is_side_view and use_displacement)):
                # Going down
                self.pushup_stage = 'down'
            elif (self.pushup_stage == 'down' and
                  (smooth_angle > threshold_up or (self.is_side_view and use_displacement)) and
                  self.frames_since_last_rep >= self.min_frames_between_reps):
                # Coming up - completing a rep
                self.pushup_count += 1
                self.pushup_stage = 'up'
                self.frames_since_last_rep = 0
        
        except Exception as e:
            st.sidebar.error(f"Angle calculation error: {str(e)}")
        
        return self.pushup_count

def process_video_enhanced(video_path, counter_obj, confidence_threshold, view_type):
    """Enhanced video processing with better side-view support"""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file!")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.info(f"Video info: {total_frames} frames at {fps:.1f} FPS")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Video display placeholder
    video_placeholder = st.empty()
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
            # Detect poses and get the annotated frame
        keypoints_data_list, annotated_frame = counter_obj.detect_pushup_pose(frame, confidence_threshold)
        
        # Count push-ups using enhanced method
        pushup_count = counter_obj.count_pushup_enhanced(keypoints_data_list)
        st.session_state.current_pushup_count = pushup_count
        
        # Display frame (every Nth frame to avoid overloading)
        if frame_count % 3 == 0:
            video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
        
        # Update progress
        progress = (frame_count + 1) / total_frames 
        progress_bar.progress(progress)
        status_text.text(f"Processing... {frame_count+1}/{total_frames} frames | Push-ups: {pushup_count}")
        
        frame_count += 1
    
    cap.release()
    
    # Save session to history
    session_data = {
        'timestamp': datetime.now(),
        'pushups': st.session_state.current_pushup_count
    }
    st.session_state.pushup_history.append(session_data)
    
    # Completion message
    st.success(f"🎉 Video processing complete! You did {st.session_state.current_pushup_count} push-ups!")
    st.balloons()
    
    # Clean up temporary video file
    os.unlink(video_path)
    
    # Clean up temporary model file if it exists
    if st.session_state.uploaded_model_temp_path and os.path.exists(st.session_state.uploaded_model_temp_path):
        os.unlink(st.session_state.uploaded_model_temp_path)
        st.session_state.uploaded_model_temp_path = None
    
    # Clear the counter object from session state
    if 'counter_obj' in st.session_state:
        del st.session_state.counter_obj

def main():
    # Header
    st.markdown('<h1 class="main-header">💪 AI Push-up Counter</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Settings")
        
        # Model upload
        st.markdown("### 📁 Upload Your YOLO Pose Model")
        model_file = st.file_uploader(
            "Upload your trained YOLO Pose model (.pt file)",
            type=['pt'],
            help="Upload your trained YOLO Pose model (e.g., yolov11m-pose.pt) for push-up detection"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
        
        # View type selection
        st.markdown("### 📷 Camera View Type")
        view_type = st.radio(
            "Select the camera angle of your video",
            ["Auto-detect", "Side View", "Front/Back View"],
            help="Help the system better detect push-ups based on camera angle"
        )
        
        # Angle threshold adjustment
        st.markdown("### ⚙️ Angle Thresholds")
        
        if view_type == "Side View" or view_type == "Auto-detect":
            down_threshold = st.slider(
                "Side View - Down Position Threshold (degrees)",
                min_value=60,
                max_value=120,
                value=100,
                step=5,
                help="Angle threshold for the bottom of push-up in side view"
            )
            
            up_threshold = st.slider(
                "Side View - Up Position Threshold (degrees)",
                min_value=130,
                max_value=170,
                value=160,
                step=5,
                help="Angle threshold for the top of push-up in side view"
            )
        else:
            down_threshold = st.slider(
                "Front/Back - Down Position Threshold (degrees)",
                min_value=60,
                max_value=120,
                value=90,
                step=5,
                help="Angle threshold for the bottom of push-up"
            )
            
            up_threshold = st.slider(
                "Front/Back - Up Position Threshold (degrees)",
                min_value=130,
                max_value=170,
                value=140,
                step=5,
                help="Angle threshold for the top of push-up"
            )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            min_vertical_displacement = st.slider(
                "Minimum Vertical Displacement (pixels)",
                min_value=10,
                max_value=100,
                value=20,
                step=5,
                help="Minimum hip movement for side view detection"
            )
            
            min_frames_between = st.slider(
                "Minimum Frames Between Reps",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                help="Prevents double counting"
            )
        
        # Update thresholds if counter object exists
        if 'counter_obj' in st.session_state and st.session_state.counter_obj:
            if view_type == "Side View":
                st.session_state.counter_obj.SIDE_VIEW_THRESHOLD_DOWN = down_threshold
                st.session_state.counter_obj.SIDE_VIEW_THRESHOLD_UP = up_threshold
            else:
                st.session_state.counter_obj.PUSHUP_THRESHOLD_DOWN = down_threshold
                st.session_state.counter_obj.PUSHUP_THRESHOLD_UP = up_threshold
            st.session_state.counter_obj.MIN_VERTICAL_DISPLACEMENT = min_vertical_displacement
            st.session_state.counter_obj.min_frames_between_reps = min_frames_between
        
        # Reset counter
        if st.button("🔄 Reset Counter", type="secondary"):
            st.session_state.current_pushup_count = 0
            if 'counter_obj' in st.session_state:
                st.session_state.counter_obj.pushup_count = 0
                st.session_state.counter_obj.pushup_stage = None
                st.session_state.counter_obj.angle_history = []
                st.session_state.counter_obj.frames_since_last_rep = 0
                st.session_state.counter_obj.hip_position_history = []
            st.success("Counter reset!")
        
        # Session goals
        st.markdown("### 🎯 Session Goal")
        goal_pushups = st.number_input(
            "Set your push-up goal",
            min_value=1,
            max_value=1000,
            value=20,
            step=5
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📹 Upload Push-up Video")
        
        # Video upload area
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video of your push-up routine"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # Display video info
            st.success("✅ Video uploaded successfully!")
            
            # Process video button
            if st.button("🚀 Start Push-up Analysis", type="primary"):
                # Handle temporary model file
                if model_file is not None:
                    if st.session_state.uploaded_model_temp_path is None or \
                       not os.path.exists(st.session_state.uploaded_model_temp_path):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
                            tmp_model.write(model_file.read())
                            st.session_state.uploaded_model_temp_path = tmp_model.name
                else:
                    st.session_state.uploaded_model_temp_path = None
                    st.warning("No model uploaded. Pose detection will not work.")
        
                # Initialize EnhancedPushupCounter object
                if 'counter_obj' not in st.session_state or \
                   (st.session_state.counter_obj.model is None and st.session_state.uploaded_model_temp_path):
                    st.session_state.counter_obj = EnhancedPushupCounter(st.session_state.uploaded_model_temp_path)
                elif st.session_state.counter_obj.model is not None and not st.session_state.uploaded_model_temp_path:
                    st.session_state.counter_obj = EnhancedPushupCounter(None)
                
                # Apply current threshold settings
                if st.session_state.counter_obj:
                    if view_type == "Side View":
                        st.session_state.counter_obj.SIDE_VIEW_THRESHOLD_DOWN = down_threshold
                        st.session_state.counter_obj.SIDE_VIEW_THRESHOLD_UP = up_threshold
                    else:
                        st.session_state.counter_obj.PUSHUP_THRESHOLD_DOWN = down_threshold
                        st.session_state.counter_obj.PUSHUP_THRESHOLD_UP = up_threshold
                    st.session_state.counter_obj.MIN_VERTICAL_DISPLACEMENT = min_vertical_displacement
                    st.session_state.counter_obj.min_frames_between_reps = min_frames_between
                
                # Reset counter state for new analysis
                st.session_state.current_pushup_count = 0
                st.session_state.counter_obj.pushup_count = 0
                st.session_state.counter_obj.pushup_stage = None
                st.session_state.counter_obj.angle_history = []
                st.session_state.counter_obj.frames_since_last_rep = 0
                st.session_state.counter_obj.hip_position_history = []
                
                process_video_enhanced(video_path, st.session_state.counter_obj, 
                                       confidence_threshold, view_type)
            
            # Display uploaded video
            st.video(uploaded_video)
    
    with col2:
        st.markdown("## 📊 Push-up Counter")
        
        # Display current count in a large card
        current_count = st.session_state.current_pushup_count
        goal_pushups_display = goal_pushups if goal_pushups > 0 else 1
        progress_percentage = min(100, (current_count / goal_pushups_display) * 100)
        
        st.markdown(f"""
        <div class="pushup-card">
            <h2>Push-ups Completed 💪</h2>
            <div class="counter-display">{current_count}</div>
            <p>Goal: {goal_pushups} push-ups</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar towards goal
        st.progress(progress_percentage / 100)
        st.write(f"Progress: {progress_percentage:.1f}% of goal")
        
        # Achievement badges
        if current_count >= goal_pushups and goal_pushups > 0:
            st.success("🎉 Goal Achieved! Great job!")
            st.balloons()
        elif current_count >= goal_pushups * 0.75 and goal_pushups > 0:
            st.info("🔥 Almost there! Keep going!")
        elif current_count >= goal_pushups * 0.5 and goal_pushups > 0:
            st.info("💪 Halfway to your goal!")
        
        # Session stats
        st.markdown("## 📈 Session Stats")
        if current_count > 0:
            estimated_calories = current_count * 0.5
            st.metric("Estimated Calories Burned", f"{estimated_calories:.1f}")
            
            # Create a simple progress chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_count,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Push-ups"},
                gauge = {
                    'axis': {'range': [None, goal_pushups_display]},
                    'bar': {'color': "#FF6B6B"},
                    'steps': [
                        {'range': [0, goal_pushups_display * 0.5], 'color': "lightgray"},
                        {'range': [goal_pushups_display * 0.5, goal_pushups_display * 0.75], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': goal_pushups_display
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Push-up History Section
        if st.session_state.pushup_history:
            st.markdown("## 📊 Push-up History")
            
            # Convert history to DataFrame
            df = pd.DataFrame(st.session_state.pushup_history)
            
            # Create timeline chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['pushups'],
                mode='lines+markers',
                name='Push-ups',
                line=dict(width=3, color='#FF6B6B'),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Push-up Progress Over Time",
                xaxis_title="Session",
                yaxis_title="Number of Push-ups",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sessions", len(df))
            with col2:
                st.metric("Total Push-ups", df['pushups'].sum())
            with col3:
                st.metric("Average per Session", f"{df['pushups'].mean():.1f}")
            
            # Display history table
            st.dataframe(df, use_column_width=True)

    # Instructions and Tips
    with st.expander("📋 How to Use This Enhanced Push-up Counter"):
        st.markdown('''
        ### Steps to Get Started:
        
        1. **Upload Your Model**: Upload your pre-trained YOLO Pose model (.pt file like `yolov11m-pose.pt`) in the sidebar.
        2. **Select Camera View**: Choose the camera angle type or let the system auto-detect.
        3. **Adjust Thresholds**: Fine-tune angle thresholds based on your video perspective.
        4. **Set Your Goal**: Choose how many push-ups you want to aim for in this session.
        5. **Upload Video**: Choose a video file of your push-up routine.
        6. **Start Analysis**: Click the "Start Push-up Analysis" button.
        7. **Monitor Debug Info**: Watch the sidebar for live angle values and detection info.
        
        ### Enhanced Features for Side-View Detection:
        
        - **Auto View Detection**: Automatically detects if the video is from side or front/back view
        - **Side-View Specific Logic**: Uses single-arm tracking when in side view
        - **Hip Displacement Tracking**: Monitors vertical hip movement for         - **Hip Displacement Tracking**: Monitors vertical hip movement for better accuracy
        - **Adaptive Thresholds**: Different angle thresholds for different viewing angles
        - **Body Alignment Analysis**: Tracks overall body posture during push-ups
        
        ### Recommended Settings by Camera Angle:
        
        #### Side View (Profile):
        - Down Position: 100-110°
        - Up Position: 150-160°
        - Min Vertical Displacement: 20-30 pixels
        
        #### Front/Back View:
        - Down Position: 85-95°
        - Up Position: 135-145°
        
        #### 45-Degree Angle:
        - Down Position: 90-100°
        - Up Position: 140-150°
        
        ### Troubleshooting Tips:
        
        - **Missing push-ups in side view**: Lower the angle thresholds or increase vertical displacement sensitivity
        - **False positives**: Increase minimum frames between reps
        - **Poor keypoint detection**: Ensure good lighting and clear visibility of the person
        - **Inconsistent counting**: Use the smoothing feature by keeping default window size
        
        ### Advanced Tips:
        
        - For side-view videos, ensure the person's profile is clearly visible
        - The system tracks hip movement in side view for better accuracy
        - Adjust thresholds based on the person's form and flexibility
        - Use higher confidence thresholds for crowded scenes
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Made with ❤️ using Streamlit and YOLO | Enhanced for better side-view detection</p>
        </div>
        """,
        unsafe_allow_html=True
    ''')

if __name__ == "__main__":
    main()
