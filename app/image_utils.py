import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

# Configuration
IMAGE_SIZE = 224  # EfficientNet-B0 input size
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_image(image_bytes):
    """
    Validate image file
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check file size
    if len(image_bytes) > MAX_FILE_SIZE:
        return False, f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.1f}MB"
    
    # Try to open image
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check if it's a valid image
        image.verify()
        
        # Check format
        if image.format not in ['JPEG', 'PNG', 'JPG']:
            return False, "Invalid image format. Only JPEG, JPG, and PNG are supported."
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def process_image(image_bytes):
    """
    Process image for model prediction
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        numpy array: Preprocessed image ready for model input (1, 224, 224, 3)
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
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
    """
    Extract FFT features from image (optional - if your model uses FFT)
    
    Args:
        image_array: Image array (224, 224, 3)
        
    Returns:
        FFT features
    """
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Apply FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = np.abs(fft_shift)
        
        # Log scale for better visualization
        magnitude_spectrum = np.log(magnitude_spectrum + 1)
        
        return magnitude_spectrum
        
    except Exception as e:
        logger.error(f"Error extracting FFT features: {str(e)}")
        raise

def preprocess_with_fft(image_bytes):
    """
    Process image with FFT features (if your model architecture requires it)
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Combined features for model input
    """
    try:
        # Get base processed image
        image_array = process_image(image_bytes)
        
        # Remove batch dimension for FFT processing
        img_single = image_array[0]
        
        # Extract FFT features
        fft_features = extract_fft_features(img_single)
        
        # Normalize FFT features
        fft_features = (fft_features - np.min(fft_features)) / (np.max(fft_features) - np.min(fft_features))
        
        # Resize FFT to match image size if needed
        fft_resized = cv2.resize(fft_features, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Stack as additional channel or concatenate based on your model architecture
        # This is just an example - adjust based on how your model was trained
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error in FFT preprocessing: {str(e)}")
        raise

def apply_augmentation(image_array):
    """
    Apply data augmentation (optional - for testing robustness)
    
    Args:
        image_array: Image array
        
    Returns:
        Augmented image array
    """
    # This is optional and typically used during training
    # Can be used to test model robustness during inference
    
    return image_array