import gradio as gr
import subprocess
from PIL import Image
import os

# Define the directory path where images will be saved (same folder as the script)
output_dir = os.getcwd()  # Get the current working directory

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

def capture_image():
    # Capture the image using libcamera-still command
    output_path = os.path.join(output_dir, "uploaded_image.jpg")
    subprocess.run(
    ["libcamera-still", "-o", output_path, "-t", "2000"],
    check=True
    )
    # Load the captured image
    img = Image.open(output_path)
    return img

def redirect_to_next_page():
    # Return HTML that includes JavaScript to redirect the page
    return gr.HTML("<meta http-equiv='refresh' content='0; url=http://192.168.137.52:5000/'>")

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Column():
        # Title at the top
        gr.Markdown("# Image Capture Interface")
        
        # Image display area in the middle
        image_display = gr.Image(type="pil", label="Captured Image")
        
        # Row to contain the buttons side by side
        with gr.Row():
            # Two buttons beside each other
            generate_button = gr.Button("Take Picture")
            next_button = gr.Button("Submit Image")
        
        # Set the button actions
        generate_button.click(capture_image, outputs=image_display)
        
        # Trigger redirection when the "Next" button is clicked
        next_button.click(redirect_to_next_page, outputs=gr.HTML())


    # Launch the Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7860)

