import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:7860"
TEST_IMAGE_PATH = "/Artificial Intelligence/imageDetection/Dataset/train/ai/imgAI602.jpg" 

def test_health_check():
    print("\n" + "="*50)
    print("Testing Health Check Endpoint")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_model_info():
    print("\n" + "="*50)
    print("Testing Model Info Endpoint")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_prediction(image_path):
    print("\n" + "="*50)
    print("Testing Prediction Endpoint")
    print("="*50)
    
    if not Path(image_path).exists():
        print(f"Error: Test image not found at {image_path}")
        print("Please update TEST_IMAGE_PATH in this script")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nPrediction Results:")
            print(f"  - Prediction: {result['prediction']}")
            print(f"  - Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            print(f"  - Real Probability: {result['probabilities']['real']:.4f}")
            print(f"  - AI Probability: {result['probabilities']['ai']:.4f}")
            print(f"  - Filename: {result['filename']}")
        else:
            print(f"Error Response: {response.json()}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_invalid_file():
    print("\n" + "="*50)
    print("Testing Invalid File Handling")
    print("="*50)
    
    try:
        dummy_content = b"This is not an image"
        files = {'file': ('test.txt', dummy_content, 'text/plain')}
        response = requests.post(f"{BASE_URL}/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.status_code == 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def run_all_tests():
    print("\n" + "="*70)
    print(" " * 20 + "ART DETECTION API TESTS")
    print("="*70)
    
    results = {
        "Health Check": test_health_check(),
        "Model Info": test_model_info(),
        "Invalid File": test_invalid_file(),
    }
    
    if Path(TEST_IMAGE_PATH).exists():
        results["Prediction"] = test_prediction(TEST_IMAGE_PATH)
    else:
        print(f"\nSkipping prediction test - no test image at {TEST_IMAGE_PATH}")
    
    print("\n" + "="*70)
    print(" " * 25 + "TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:.<50} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)

if __name__ == "__main__":
    print("\nMake sure the server is running on http://localhost:7860")
    print("Start server with: python app/main.py\n")
    
    run_all_tests()