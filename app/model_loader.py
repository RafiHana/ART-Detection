import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.model_path = self._get_model_path()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.backends.mps.is_available():
             self.device = torch.device("mps")
        
        self.class_names = ['ai', 'real'] 
        
    def _get_model_path(self):
        current_dir = Path(__file__).parent
        model_dir = current_dir.parent / "models"
        
        possible_names = ["finalModel.pth", "bestModel.pth"]
        
        for name in possible_names:
            model_path = model_dir / name
            if model_path.exists():
                return model_path
        
        return model_dir / possible_names[0]
    
    def load_model(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            logger.info(f"Loading PyTorch model from: {self.model_path}")
            
            self.model = models.efficientnet_b0(weights=None) 
            
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_ftrs, 2)
            )
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info("Model loaded successfully on device: " + str(self.device))
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            raise

    def predict(self, image_tensor):
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        try:
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad(): 
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                probs_list = probabilities[0].tolist() 
                
                ai_prob = probs_list[0]   
                real_prob = probs_list[1] 
                
                if real_prob > ai_prob:
                    prediction_class = 'real'
                    confidence = real_prob
                else:
                    prediction_class = 'ai'
                    confidence = ai_prob
                
                return prediction_class, confidence, {'real': real_prob, 'ai': ai_prob}
                
        except Exception as e:
            logger.error(f"Error prediction: {str(e)}")
            raise