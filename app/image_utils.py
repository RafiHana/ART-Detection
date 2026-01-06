import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024 

def validate_image(image_bytes):
    if len(image_bytes) > MAX_FILE_SIZE:
        return False, f"File size too large"
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        if image.format not in ['JPEG', 'PNG', 'JPG', 'WEBP']:
             return False, "Invalid format"
        return True, None
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

class FFTTransformProcess:
    def process(self, img):
        # 1. Grayscale
        img_gray = img.convert('L')
        img_array = np.array(img_gray)

        # 2. FFT
        f = np.fft.fft2(img_array)
        fshift = np.fft.fftshift(f)

        # 3. Magnitude & Log
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # 4. Normalize 0-255
        magnitude_spectrum = np.nan_to_num(magnitude_spectrum)
        ms_min = np.min(magnitude_spectrum)
        ms_max = np.max(magnitude_spectrum)

        if ms_max - ms_min > 0:
            img_fft = 255 * (magnitude_spectrum - ms_min) / (ms_max - ms_min)
        else:
            img_fft = np.zeros_like(magnitude_spectrum)

        img_fft = img_fft.astype(np.uint8)

        # 5. Back to RGB (3 Channel)
        img_fft_pil = Image.fromarray(img_fft).convert("RGB")
        return img_fft_pil

def process_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        resizer = transforms.Resize((224, 224))
        image_resized = resizer(image)
        
        #Apply FFT
        fft_processor = FFTTransformProcess()
        image_fft = fft_processor.process(image_resized)
        
        #ToTensor & Normalize
        transform_final = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform_final(image_fft)
        
        input_tensor = input_tensor.unsqueeze(0)
        
        logger.info(f"Processed tensor shape: {input_tensor.shape}")
        return input_tensor
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise