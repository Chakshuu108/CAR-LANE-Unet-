import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import cv2
import numpy as np
import tempfile

# ========================================
# CONFIG
# ========================================
IMG_HEIGHT = 256
IMG_WIDTH = 512
CONFIDENCE_THRESHOLD = 0.5
MAX_VIDEO_WIDTH = 1280
MAX_VIDEO_HEIGHT = 720

# ========================================
# LOAD MODEL (Lazy - only when needed)
# ========================================
@st.cache_resource
def load_model():
    try:
        from tensorflow import keras
        model_path = 'best_lane_model_stage2.h5'
        if not os.path.exists(model_path):
            st.error("❌ Model file not found: best_lane_model_stage2.h5")
            st.stop()
        
        with st.spinner("Loading model..."):
            model = keras.models.load_model(model_path)
        return model
    except ImportError:
        st.error("TensorFlow not installed. Install locally to use full features.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# ========================================
# FUNCTIONS
# ========================================

def resize_frame(frame, max_width=1280, max_height=720):
    """Resize frame maintaining aspect ratio"""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    
    if width > max_width or height > max_height:
        if aspect_ratio > (max_width / max_height):
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return frame


def predict_lane_mask(image_array, model):
    """Predict lane mask from image array"""
    original_image = image_array.copy()
    original_height, original_width = image_array.shape[:2]
    
    # Preprocess
    image_resized = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    # Predict
    prediction = model.predict(image_batch, verbose=0)
    predicted_mask = prediction[0].squeeze()
    
    # Threshold
    predicted_mask_binary = (predicted_mask > CONFIDENCE_THRESHOLD).astype(np.uint8)
    
    # Resize back to original size
    predicted_mask_resized = cv2.resize(
        predicted_mask_binary.astype(np.float32),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )
    
    return original_image, predicted_mask_resized


def overlay_lanes_green(image, mask, alpha=0.7):
    """Apply green colored transparent overlay on detected lanes"""
    overlay = image.copy().astype(np.float32)
    
    green_color = (0, 255, 0)
    mask_colored = np.zeros_like(image, dtype=np.float32)
    
    for i in range(3):
        mask_colored[:, :, i] = mask * green_color[i]
    
    result = image.astype(np.float32)
    result[mask > 0] = (1 - alpha) * result[mask > 0] + alpha * mask_colored[mask > 0]
    
    return result.astype(np.uint8)


def process_video(video_path, model, progress_bar, status_text, alpha):
    """Process video with lane detection and return output path"""
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        fps = 30
    
    # Calculate dimensions
    aspect_ratio = width / height
    if width > MAX_VIDEO_WIDTH or height > MAX_VIDEO_HEIGHT:
        if aspect_ratio > (MAX_VIDEO_WIDTH / MAX_VIDEO_HEIGHT):
            new_width = MAX_VIDEO_WIDTH
            new_height = int(MAX_VIDEO_WIDTH / aspect_ratio)
        else:
            new_height = MAX_VIDEO_HEIGHT
            new_width = int(MAX_VIDEO_HEIGHT * aspect_ratio)
    else:
        new_width = width
        new_height = height
    
    # Make dimensions even
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1
    
    # Output file
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = output_temp.name
    output_temp.close()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    frame_count = 0
    
    # Process frames
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Resize frame
        frame_resized = resize_frame(frame, MAX_VIDEO_WIDTH, MAX_VIDEO_HEIGHT)
        
        # Ensure exact dimensions
        if frame_resized.shape[1] != new_width or frame_resized.shape[0] != new_height:
            frame_resized = cv2.resize(frame_resized, (new_width, new_height))
        
        # Convert BGR to RGB for model
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Predict lane mask
        original_image, predicted_mask = predict_lane_mask(frame_rgb, model)
        
        # Apply green overlay
        result_rgb = overlay_lanes_green(original_image, predicted_mask, alpha=alpha)
        
        # Convert back to BGR for saving
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(result_bgr)
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(min(progress, 0.99))
        status_text.text(f"Processing: {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    return output_path, new_width, new_height, fps, frame_count


# ========================================
# STREAMLIT UI
# ========================================

st.set_page_config(
    page_title="Lane Detection with AI",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ Lane Detection with AI")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Instructions")
    st.markdown("""
    1. Upload an MP4 video file
    2. Adjust the overlay opacity if needed
    3. Click 'Process Video' to apply lane detection
    4. Download the processed video with green lane overlay
    
    **Features:**
    - ✅ Real-time lane detection
    - ✅ Green lane overlay
    - ✅ HD quality output
    - ✅ One-click download
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    alpha = st.slider("Overlay Opacity", 0.3, 1.0, 0.7, 0.1)
    st.info(f"Current opacity: {alpha:.1%}")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose an MP4 video file",
        type=["mp4"],
        help="Upload a video file (MP4 format)"
    )

with col2:
    st.markdown("### 📊 Video Info")
    video_info = st.empty()

# Process uploaded file
if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name
    
    try:
        # Get video info
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            fps = 30
        
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Display video info
        with video_info:
            col1_info, col2_info, col3_info, col4_info = st.columns(4)
            with col1_info:
                st.metric("Resolution", f"{width}x{height}")
            with col2_info:
                st.metric("Frames", f"{total_frames}")
            with col3_info:
                st.metric("Duration", f"{duration:.1f}s")
            with col4_info:
                st.metric("FPS", f"{fps:.1f}")
        
        # Process button
        col1, col2, col3 = st.columns(3)
        
        with col2:
            process_button = st.button(
                "🚀 Process Video",
                use_container_width=True,
                type="primary"
            )
        
        if process_button:
            # Load model
            model = load_model()
            
            # Create placeholders for progress
            st.markdown("---")
            st.markdown("### ⏳ Processing...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process video
            result = process_video(temp_video_path, model, progress_bar, status_text, alpha)
            
            if result:
                output_path, new_width, new_height, output_fps, processed_frames = result
                
                # Success message
                status_text.empty()
                progress_bar.empty()
                
                st.success("✅ Video processing completed!")
                
                # Show results
                st.markdown("---")
                st.markdown("### 📥 Download Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric("Output Resolution", f"{new_width}x{new_height}")
                    st.metric("Frames Processed", f"{processed_frames}")
                
                with result_col2:
                    st.metric("Output FPS", f"{output_fps:.1f}")
                    try:
                        file_size = os.path.getsize(output_path) / (1024 * 1024)
                        st.metric("File Size", f"{file_size:.2f} MB")
                    except:
                        st.metric("File Size", "Unknown")
                
                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f.read(),
                        file_name="lane_detection_output.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
                # Cleanup
                try:
                    os.unlink(output_path)
                except:
                    pass
            else:
                st.error("Error processing video")
        
        # Cleanup temp file
        try:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        except:
            pass
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        try:
            os.unlink(temp_video_path)
        except:
            pass

else:
    st.info("👆 Please upload a video file to get started")
