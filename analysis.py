import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2, collections
import pandas as pd
from deepface import DeepFace as deepface
from my_functions import concentration_index, gaze_analysis, map_emotion_to_AU, process_openface_gaze, compare_gaze_estimations, plot_au_over_time
import matplotlib.pyplot as plt

video_capture = cv2.VideoCapture("./prerecorded.mp4")
fps = video_capture.get(cv2.CAP_PROP_FPS)      
frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
duration = int(frame_count/fps)
WINDOW_SIZE = 60
OVERLAP = int(WINDOW_SIZE * 0.5)
deepface_results = []
gaze_data = []
frame_number = 0
concentration_values = []
avg_concentration_values = []
emotion_common = []

emotion_buffer = collections.deque(maxlen=WINDOW_SIZE + 1)
gaze_buffer = collections.deque(maxlen=WINDOW_SIZE + 1)
concentration_buffer = collections.deque(maxlen=WINDOW_SIZE + 1)


while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        try:
            # Detect face and extract features using DeepFace
            detections = deepface.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend="skip")
            if detections:
                bbox = detections[0]['region']
                emotion = detections[0]['dominant_emotion']
                emotion_buffer.append(emotion)
                
                deepface_results.append({
                "frame": len(deepface_results) + 1,
                "emotion": emotion
                 })

                # Draw bounding box
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Gaze analysis
            gaze_direction, gaze_weights, gaze_data = gaze_analysis(frame, gaze_data)
            gaze_buffer.append(gaze_weights)

            # Calculate concentration index
            concentration = concentration_index(emotion, gaze_weights)
            concentration_values.append(concentration)

            # Display concentration message based on index
            concentration_buffer.append(concentration)
            
            if len(emotion_buffer) == WINDOW_SIZE or frame_number == frame_count - 1:
                    # Aggregate emotions (majority voting)
                    most_common_emotion = collections.Counter(emotion_buffer).most_common(1)[0][0]
                    # Display emotion on the frame
                    cv2.putText(frame, f"Emotion: {most_common_emotion}",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if len(gaze_buffer) > 0:
                        sorted_gaze = sorted(gaze_buffer)
                        length = len(sorted_gaze)
                        if length % 2 == 0:
                            median = (sorted_gaze[length // 2 - 1] + sorted_gaze[length // 2]) / 2
                        else:
                            median = sorted_gaze[length // 2]
                    else:
                        median = 0  # Default value if gaze_buffer is empty

                    # Average concentration index
                    emotion_common.append({"emotion": most_common_emotion})
                    avg_concentration = concentration_index(most_common_emotion, median)
                    avg_concentration_values.append(avg_concentration)
                    
                    #Cancel the first OVERLAP elements
                    for _ in range(OVERLAP):
                        if emotion_buffer:
                            emotion_buffer.popleft()
                        if gaze_buffer:
                            gaze_buffer.popleft()
                        if concentration_buffer:
                            concentration_buffer.popleft()


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
            
            frame_number += 1
            
            cv2.imshow("Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Press 'q' to exit the loop
            
        except Exception as e:
            print(f"Error processing frame: {e}")          

deepface_df = pd.DataFrame(deepface_results)
deepface_df.to_csv("deepface_emotions.csv", index=False)
deepface_df = pd.read_csv("deepface_emotions.csv")

#CLEAN            
video_capture.release()
cv2.destroyAllWindows()
    
dir_path = os.getcwd() + '\\OpenFace_2.2.0_win_x64\\processed'
file_path = os.path.join(dir_path, 'prerecorded.csv')
#au_nnz, au_int = process_au(file_path)
#plot_hist(au_nnz, "anger")
#plot_au_int(au_int, "prerecorded")

plot_au_over_time(dir_path, "prerecorded")

#Deepface
# Load OpenFace AU data
openface_df = pd.read_csv(file_path)

# Convert OpenFace AU columns to numeric
au_columns = [col for col in openface_df.columns if col.endswith('_c')]

# Apply AU mapping to DeepFace emotions and expand into separate columns
deepface_au_df = deepface_df["emotion"].apply(map_emotion_to_AU).apply(pd.Series)
deepface_au_df = deepface_au_df[au_columns]
deepface_au_df.to_csv("deepface_emotions_au.csv", index=False)

# Compute correlation between OpenFace AU classifications and DeepFace AU mappings
plot_au_over_time(os.getcwd(), "deepface_emotions_au")
correlations = openface_df[au_columns].corrwith(deepface_au_df)
au_45 = openface_df[au_columns].iloc[:, -1]

# Plot correlation
plt.figure(figsize=(10, 6))
correlations.plot(kind="bar", color="orange")
plt.title("Correlation between DeepFace Emotion-based AUs and OpenFace AUs")
plt.xlabel("Action Units (AUs)")
plt.ylabel("Correlation Coefficient")
plt.xticks(rotation=90)
plt.show()

#GAZE
gaze_of_x, gaze_of_y = process_openface_gaze(file_path)
compare_gaze_estimations(gaze_of_x, gaze_of_y, au_45, gaze_data)

#CONCENTRATION
plt.figure(figsize=(10, 5))
plt.plot(concentration_values, marker='o', linestyle='-', color='b', label="Concentration Level")
plt.axhline(y=0.65, color='g', linestyle='--', label="Engaged Threshold")
plt.axhline(y=0.25, color='r', linestyle='--', label="Low Concentration Threshold")

plt.xlabel("Frame Number")
plt.ylabel("Concentration Index")
plt.title("Concentration Level Over Time")
plt.legend()
plt.grid(True)
plt.show()

#CONCENTRATION BUFFER
plt.figure(figsize=(10, 5))
plt.plot(avg_concentration_values, marker='o', linestyle='-', color='b', label="Concentration Level")
plt.axhline(y=0.65, color='g', linestyle='--', label="Engaged Threshold")
plt.axhline(y=0.25, color='r', linestyle='--', label="Low Concentration Threshold")

plt.xlabel("Frame Number")
plt.ylabel("Concentration Index")
plt.title("Concentration Level Over Time, buffer")
plt.legend()
plt.grid(True)
plt.show()

#ANALYSIS ON TIME WINDOWS
openface_df = openface_df.sort_values(by="frame")

# Initialize aggregated data list
rolling_aggregated = []

# Iterate with step size accounting for overlap
for start_frame in range(0, openface_df['frame'].max(), OVERLAP):
    end_frame = start_frame + WINDOW_SIZE
    window_data = openface_df[(openface_df['frame'] >= start_frame) & (openface_df['frame'] < end_frame)]

    if not window_data.empty:
        # Aggregate Action Units (AUs) and gaze within this time window
        aggregated_values = window_data[[
            ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', 
            ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', 
            ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' gaze_angle_x', 
            ' gaze_angle_y', ' AU45_c'
        ]].median()

        # Store results
        rolling_aggregated.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "time_window": start_frame / 30,  # Convert to seconds
            **aggregated_values
        })

# Convert to DataFrame
aggregated_df = pd.DataFrame(rolling_aggregated)

# Save processed data
aggregated_df.to_csv("openface_aggregated.csv", index=False)
au_columns = [col for col in aggregated_df.columns if col.endswith('_c')]
aggregated_df[au_columns] = aggregated_df[au_columns].astype(int)

deepface_common_au = pd.DataFrame(emotion_common)
deepface_common_au = deepface_common_au["emotion"].apply(map_emotion_to_AU).apply(pd.Series)
deepface_common_au.to_csv("deepface_timewindow.csv", index=False)
deepface_common_au = pd.read_csv("deepface_timewindow.csv")
deepface_common_au = deepface_common_au[au_columns]


aggregated_df = aggregated_df.iloc[:-1]  
plot_au_over_time(os.getcwd(), "openface_aggregated")
deepface_common_au.to_csv("deepface_timewindow.csv", index=False)
plot_au_over_time(os.getcwd(), "deepface_timewindow")
correlations = aggregated_df[au_columns].corrwith(deepface_common_au)
au_45_agg = aggregated_df[au_columns].iloc[:, -1]

# Plot correlation
plt.figure(figsize=(10, 6))
correlations.plot(kind="bar", color="orange")
plt.title("Correlation between DeepFace Emotion-based AUs and OpenFace AUs with Time windows")
plt.xlabel("Action Units (AUs)")
plt.ylabel("Correlation Coefficient")
plt.xticks(rotation=90)
plt.show()