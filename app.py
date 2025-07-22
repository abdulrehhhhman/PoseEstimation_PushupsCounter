import streamlit as st
import pickle
import tempfile
import cv2
import os
from pose_counter import process_video_enhanced

st.set_page_config(page_title="Push-Up Counter", layout="centered")
st.title("🏋️ Push-Up Counter App")
st.markdown("Upload a video to count the number of push-ups performed.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_file.write(uploaded_file.read())
    video_path = temp_video_file.name

    st.video(video_path)

    st.info("Processing video... please wait.")

    # Load keypoints
    pickle_path = video_path.replace(".mp4", ".pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            keypoints_data = pickle.load(f)

        pushup_count = process_video_enhanced(keypoints_data)
        st.success(f"✅ Total Push-Ups Counted: **{pushup_count}**")
    else:
        st.error("❌ Keypoint data (.pkl) not found. Please upload preprocessed video.")
