import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import kagglehub
from deepface import DeepFace as deepface
import cv2
import json
import subprocess
import re

# Emotion label mapping
emotion_mapping = {
    'angry': 'angry',
    'disgusted': 'disgust',
    'fearful': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'surprised': 'surprise'
}

au_mapping = {
    'happy': {'AU06_c': 1, 'AU12_c': 1, 'AU25_c': 1},
    'angry': {'AU04_c': 1, 'AU05_c': 1, 'AU07_c': 1, 'AU10_c': 1, 'AU17_c': 1, 'AU23_c': 1, 'AU25_c': 1, 'AU26_c': 1},
    'sad': {'AU01_c': 1, 'AU04_c': 1, 'AU06_c': 1, 'AU15_c': 1, 'AU17_c': 1},
    'surprise': {'AU01_c': 1, 'AU02_c': 1, 'AU05_c': 1, 'AU26_c': 1},
    'fear': {'AU01_c': 1, 'AU02_c': 1, 'AU04_c': 1, 'AU05_c': 1, 'AU20_c': 1, 'AU25_c': 1, 'AU26_c': 1},
    'disgust': {'AU09_c': 1, 'AU10_c': 1, 'AU17_c': 1, 'AU25_c': 1, 'AU26_c': 1},
    'neutral': {}
}

def predict_emotion(au_output):
    for emotion, aus in au_mapping.items():
        if all(au_output.get(au, 0) == val for au, val in aus.items()):
            return emotion
    return 'unknown'

def parse_au_output(output):
    au_output = {}
    for line in output.split('\n'):
        match = re.search(r'(AU\d{2}_c)\s*:\s*(\d)', line)
        if match:
            au_output[match.group(1)] = int(match.group(2))
    return au_output

# Download dataset
path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")
dataset_path = os.path.join(path, "test")

# Load DeepFace for emotion detection
results = []
correct_predictions = 0
total_images = 0

for directory in os.listdir(dataset_path):
    expected_emotion = emotion_mapping.get(directory.lower(), directory.lower())
    print(expected_emotion)
    for img_file in os.listdir(os.path.join(dataset_path, directory)):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(dataset_path, directory, img_file)
            
            # Run OpenFace Feature Extraction
            output = subprocess.run(
                [".\\OpenFace_2.2.0_win_x64\\FeatureExtraction", "-f", img_path, "-aus"],
                capture_output=True, text=True
            )
            
            au_output = parse_au_output(output.stdout)
            print(output.stdout)
            predicted_emotion = predict_emotion(au_output)

            results.append({
                'image': img_file,
                'expected_emotion': expected_emotion,
                'predicted_emotion': predicted_emotion,
                'aus': au_output
            })


            total_images += 1
            if predicted_emotion == expected_emotion:
                correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_images if total_images > 0 else 0.0
print(f"Accuracy: {accuracy:.2%}")

# Save results to JSON file
with open('emotion_detection_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Emotion detection completed. Results saved to emotion_detection_results.json")
