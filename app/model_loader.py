import tensorflow as tf
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.model_path = self._get_model_path()
        self.class_names = ['real', 'ai']
        
    def _get_model_path(self):
        current_dir = Path(__file__).parent
        model_dir = current_dir.parent / "models"
        
        possible_names = [
            "finalModel.pth"
        ]
        
        for name in possible_names:
            model_path = model_dir / name
            if model_path.exists():
                return model_path
        
        return model_dir / possible_names[0]
    
    def load_model(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. "
                    f"Please ensure your trained model is saved in the models directory."
                )
            
            logger.info(f"Loading model from: {self.model_path}")
            
            self.model = tf.keras.models.load_model(
                str(self.model_path),
                compile=False  
            )
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            raise
    
    def predict(self, image_array):
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        try:
            predictions = self.model.predict(image_array, verbose=0)
            if predictions.shape[-1] == 1:
                ai_prob = float(predictions[0][0])
                real_prob = 1.0 - ai_prob
            else:
                real_prob = float(predictions[0][0])
                ai_prob = float(predictions[0][1])
            
            if real_prob > ai_prob:
                prediction_class = 'real'
                confidence = real_prob
            else:
                prediction_class = 'ai'
                confidence = ai_prob
            
            probabilities = [real_prob, ai_prob]
            
            return prediction_class, confidence, probabilities
            
        except Exception as e:
            logger.error(f"Error prediction: {str(e)}")
            raise
    
    def get_model_summary(self):
        if not self.is_loaded:
            return "Model not loaded"
        
        return self.model.summary()