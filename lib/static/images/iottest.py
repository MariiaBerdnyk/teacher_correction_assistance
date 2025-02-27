import gradio as gr
import subprocess
from PIL import Image
import os
import threading

# Define the directory path where images will be saved (same folder as the script)
output_dir = os.getcwd()  # Get the current working directory

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Global variable to track the video preview process
video_process = None

def start_video_preview():
    """Starts the video feed using libcamera-vid."""
    global video_process
    if video_process is None:
        try:
            # Start the video feed and output it to a named pipe
            video_process = subprocess.Popen(
                ["libcamera-vid", "--nopreview", "-t", "0", "--inline", "-o", "/dev/null"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return "Video preview started. Ready to capture."
        except Exception as e:
            return f"Error starting video preview: {e}"
    else:
        return "Video preview is already running."

def stop_video_preview():
    """Stops the video feed."""
    global video_process
    if video_process:
        try:
            video_process.terminate()
            video_process.wait()
            video_process = None
            return "Video preview stopped."
        except Exception as e:
            return f"Error stopping video preview: {e}"
    else:
        return "No video preview to stop."

def capture_image():
    """Captures an image using libcamera-still."""
    global video_process
    # Stop the video feed if running
    if video_process:
        stop_video_preview()
    
    # Define the path for the output image
    output_path = os.path.join(output_dir, "captured_image.jpg")
    
    try:
        # Use libcamera-still to capture an image
        subprocess.run(
            ["libcamera-still", "--nopreview", "-o", output_path, "-t", "2000"],
            check=True
        )
        # Load the captured image using PIL
        img = Image.open(output_path)
        return img
    except Exception as e:
        return f"Error capturing image: {e}"

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Column():
        # Title at the top
        gr.Markdown("# Image Capture Interface with Video Preview")
        
        # Text area for status messages
        status = gr.Textbox(label="Status", interactive=False)
        
        # Image display area in the middle
        image_display = gr.Image(type="pil", label="Captured Image")
        
        # Buttons for starting/stopping the video feed and capturing an image
        with gr.Row():
            start_button = gr.Button("Start Video Preview")
            stop_button = gr.Button("Stop Video Preview")
            capture_button = gr.Button("Take Picture")
        
        # Set the button actions
        start_button.click(start_video_preview, outputs=status)
        stop_button.click(stop_video_preview, outputs=status)
        capture_button.click(capture_image, outputs=image_display)

    # Launch the Gradio interface on the local network
    demo.launch(server_name="0.0.0.0", server_port=7860)

