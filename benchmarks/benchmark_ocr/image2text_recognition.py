import argparse
import json
import torch
import os
import logging

# import torch_xla.core.xla_model as xm

from PIL import Image, ImageFile
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from lib.model import Model


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon devices
    else:
        device = torch.device("cpu")
    return device


class OCRModel(Model):
    def __init__(self, device: torch.device, model_path: str = "microsoft/trocr-base-handwritten"):
        super().__init__("Image2Text")
        self.device = device
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model = self.model.to(device=device)

    def extract_text(self, image: ImageFile.ImageFile|list[ImageFile.ImageFile], max_tokens: int = 200) -> str:
        """
            Extract Text from image using model
            :param image: an image or a list of image
            :param max_tokens
            :return the text extracted
        """
        images = None
        if type(image) == ImageFile:
            images = [image]
        elif type(image) == list:
            images = image
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, max_new_tokens=max_tokens).to(self.device)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def load_images_from_directory(directory):
    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
    images = []

    # Iterate through all files in the directory
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if file_name.lower().endswith(supported_formats):
            try:
                image = Image.open(file_path).convert("RGB")
                images.append(image)
                # print(f"Loaded image: {file_path}")
            except Exception as e:
                print(f"Failed to load image: {file_path}. Error: {e}")

    return images

def process_batch(images, model, batch_size=4):
    # Split the images into smaller batches to avoid OOM errors
    batch_texts = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        generated_text = model.extract_text(batch_images)

        batch_texts.extend(generated_text)

    return batch_texts

def main():
    parser = argparse.ArgumentParser(description="Process images for text recognition.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing image files.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size of images to fit into the model.")
    parser.add_argument("--device",  type=str, required=False,
                        help="The device to be used for benchmark.")
    args = parser.parse_args()

    # Load images from the specified directory
    images = load_images_from_directory(args.image_dir)

    # Get device
    if args.device is None:
        device = get_device()
    else:
        device = torch.device(args.device)

    model = OCRModel(device, args.model_name)

    # Process images in batches
    generated_text = process_batch(images, model, args.batch_size)

    print(json.dumps(generated_text, ensure_ascii=False, indent=2))  # JSON formatted output
    return generated_text

if __name__ == "__main__":
    main()
