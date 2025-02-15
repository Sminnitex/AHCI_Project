from GazeTracking.gaze_tracking import GazeTracking
from deepface import DeepFace as deepface
import numpy as np
import cv2

gaze = GazeTracking()
TIME_WINDOW_SIZE = 60
frame_counter = 0

# Video capture object
video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture("./prerecorded.mp4")

# Define emotion weights
emotion_weights = {
    'happy': 0.6,
    'angry': 0.25,
    'surprise': 0.6,
    'sad': 0.3,
    'fear': 0.3,
    'disgust': 0.2,
    'neutral': 0.9
}

def eucledian_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def calculate_eye_openess(eye):
    if eye is None:
        return 0
    
    A = eucledian_distance(eye.landmark_points[1], eye.landmark_points[5])
    B = eucledian_distance(eye.landmark_points[2], eye.landmark_points[4])
    C = eucledian_distance(eye.landmark_points[0], eye.landmark_points[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

def concentration_index(emotion, gaze_weights):
    # Get emotion weight
    emotion_score = emotion_weights.get(emotion, 0.5)  # Default to 0.5 if emotion not found

    # Calculate the concentration index as a weighted sum of emotion and gaze
    concentration = (emotion_score + gaze_weights) / 4.5  # Percentage of scores

    return concentration

def gaze_analysis(frame):
    gaze.refresh(frame)
    
    openess_left = calculate_eye_openess(gaze.eye_left)
    openess_right = calculate_eye_openess(gaze.eye_right)
    avarage_openess = (openess_left + openess_right) / 2
    
    if gaze.is_blinking():
        gaze_direction = "Blinking"
    elif gaze.is_right():
        gaze_direction = "Looking Right"
    elif gaze.is_left():
        gaze_direction = "Looking Left"
    elif gaze.is_center():
        gaze_direction = "Looking Center"
    else:
        gaze_direction = "Unknown"
        
    if avarage_openess < 0.2:  # Closed eyes
        gaze_weight = 0
    elif 0.2 <= avarage_openess < 0.3:  # Semi-closed
        gaze_weight = 1.5
    else:
        if gaze_direction == "Looking Center":
            gaze_weight = 5
        else:
            gaze_weight = 2
        
    return gaze_direction, gaze_weight

def generate_frames():
    global frame_counter
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        try:
            # Process the DeepFace analysis only in defined time windows
            if frame_counter % TIME_WINDOW_SIZE == 0:
                # Detect face and extract features using DeepFace
                detections = deepface.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend="skip")
                if detections:
                    bbox = detections[0]['region']
                    emotion = detections[0]['dominant_emotion']

                    # Draw bounding box
                    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # Gaze analysis (you can still keep this running on every frame if needed)
            gaze_direction, gaze_weights = gaze_analysis(frame)

            # Calculate concentration index based on gaze and emotion
            concentration = concentration_index(emotion, gaze_weights)

            # Display concentration message based on index
            if concentration > 0.65:
                cv2.putText(frame, "You are engaged",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif concentration > 0.25 and concentration <= 0.65:
                cv2.putText(frame, "You are pretty concentrated",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Keep attention!",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display gaze information
            cv2.putText(frame, f"Gaze: {gaze_direction}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Increment frame counter
            frame_counter += 1

        except Exception as e:
            print(f"Error processing frame: {e}")

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')