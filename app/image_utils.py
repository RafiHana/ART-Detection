import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224  
MAX_FILE_SIZE = 10 * 1024 * 1024 

def validate_image(image_bytes):
    if len(image_bytes) > MAX_FILE_SIZE:
        return False, f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.1f}MB"
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        image.verify()
        
        if image.format not in ['JPEG', 'PNG', 'JPG']:
            return False, "Invalid image format. Only JPEG, JPG, and PNG are supported."
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def process_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        logger.info(f"Processed image shape: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def extract_fft_features(image_array):
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Apply FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = np.abs(fft_shift)
        
        magnitude_spectrum = np.log(magnitude_spectrum + 1)
        
        return magnitude_spectrum
        
    except Exception as e:
        logger.error(f"Error extracting FFT features: {str(e)}")
        raise

def preprocess_with_fft(image_bytes):
    try:
        image_array = process_image(image_bytes)
        img_single = image_array[0]
        
        fft_features = extract_fft_features(img_single)
        
        fft_features = (fft_features - np.min(fft_features)) / (np.max(fft_features) - np.min(fft_features))
        
        fft_resized = cv2.resize(fft_features, (IMAGE_SIZE, IMAGE_SIZE))
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error in FFT preprocessing: {str(e)}")
        raise

def apply_augmentation(image_array):
    return image_array