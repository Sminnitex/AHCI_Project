import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from GazeTracking.gaze_tracking import GazeTracking

# Define emotion weights
emotion_weights = {
    'happy': 0.6,
    'angry': 0.25,
    'surprise': 0.6,
    'sad': 0.3,
    'fear': 0.35, #In the paper the code is actually 0.30, for the analysis ex i set con 0.35 to distinguish between emotions
    'disgust': 0.2,
    'neutral': 0.9
}

all_AUs_c = {' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', 
                ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', 
                ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c'}

gaze = GazeTracking()

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
    concentration = (emotion_score + gaze_weights) / 6.0  # Percentage of scores

    return concentration

def gaze_analysis(frame, gaze_data):
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
    
    gaze_data.append([frame, gaze_direction, avarage_openess])    
    return gaze_direction, gaze_weight, gaze_data

#ANALYSIS
def process_au(file_path):
    df = pd.read_csv(file_path)
    au_numb = 34
    num_frames = len(df)
    au_data = df.iloc[:, -au_numb:] 

    au_countnnz = (au_data.iloc[:, -au_numb:]!=0).sum()
    au_normalized_nnz = au_countnnz / num_frames
    

    return au_normalized_nnz, au_data


def plot_hist(au, file):
    plt.figure(figsize=(10, 6))
    au.plot(kind='bar', color='skyblue')
    
    plt.title(f'Normalized AU Activations - {file}')
    plt.xlabel('Action Units (AU)')
    plt.ylabel('Normalized Activation')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.yticks([0, 1, 0.2, 0.4, 0.6, 0.8])
    #plt.savefig(f'{file}_au_histogram_nnz.png')
    plt.show()
    
def plot_au_int(au_int, expression, bins=10):    
    au_names = au_int.columns.tolist() 
    
    for au_name in au_names:
        intensities = au_int[au_name]  
        
        hist, bin_edges = np.histogram(intensities, bins=np.linspace(0, 1, bins+1), density=True)
        
        plt.figure(figsize=(8, 5))
        plt.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0], color='lightcoral', edgecolor='black')
        plt.title(f'Normalized Intensity Histogram for {au_name}')
        plt.xlabel('Intensity Level (0.0 - 1.0)')
        plt.ylabel('Normalized Count')
        plt.xticks(np.linspace(0, 1, bins+1))  
        plt.tight_layout()
        #plt.savefig(f'{expression}_au_histogram_int_{au_name}.png')
        plt.show()

def plot_au_over_time(dir_path, expression_name):
    file_path = os.path.join(dir_path, f'{expression_name}.csv')
    
    # Load CSV
    df = pd.read_csv(file_path)

    # Extract relevant AUs
    au_columns = [col for col in df.columns if col in all_AUs_c]
    if not au_columns:
        print(f"No matching AUs found in {file_path}")
        return
    
    au_data = df[au_columns]

    # Normalize AU activations
    rolling_window = 10  
    au_data_normalized = au_data.rolling(window=rolling_window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Dictionary to store line objects
    lines = {}
    
    # Plot each AU over time and store the line object
    for au in au_columns:
        line, = ax.plot(df.index, au_data_normalized[au], label=au.strip(), picker=True, linewidth=1.5)
        lines[au.strip()] = line  # Store the line reference
    
    # Add legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    # Store legend entries for interaction
    legend_lines = legend.get_lines()
    highlighted_au = None  

    # Define event handler for clicking on legend
    def on_legend_click(event):
        nonlocal highlighted_au  # Access the global variable
        
        legend_line = event.artist  # Get the legend line that was clicked
        label = legend_line.get_label()  # Get the corresponding AU label
        
        if label in lines:
            if highlighted_au == label:
                # If clicked again, show all lines
                for line in lines.values():
                    line.set_visible(True)
                highlighted_au = None  # Reset
            else:
                # Hide all lines except the selected one
                for au, line in lines.items():
                    line.set_visible(au == label)
                highlighted_au = label  # Store the current highlighted AU
            
            # Refresh plot
            fig.canvas.draw()

    # Connect event handler
    fig.canvas.mpl_connect('pick_event', on_legend_click)

    # Show plot
    plt.title(f'Normalized AU Activations Over Time - {expression_name.capitalize()}')
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized AU Activation')
    plt.tight_layout()
    plt.show()
    
def map_emotion_to_AU(emotion):
    mapping = {
        'happy': {' AU06_c': 1, ' AU12_c': 1, ' AU25_c': 1},
        'angry': {' AU04_c': 1, ' AU05_c': 1, ' AU07_c': 1, ' AU10_c': 1, ' AU17_c': 1, 
                  ' AU23_c': 1, ' AU25_c': 1, ' AU26_c': 1},
        'sad': {' AU01_c': 1, ' AU04_c': 1, ' AU06_c': 1, ' AU15_c': 1, ' AU17_c': 1},
        'surprise': {' AU01_c': 1, ' AU02_c': 1, ' AU05_c': 1, ' AU26_c': 1},
        'fear': {' AU01_c': 1, ' AU02_c': 1, ' AU04_c': 1, ' AU05_c': 1, ' AU20_c': 1, 
                 ' AU25_c': 1, ' AU26_c': 1},
        'disgust': {' AU09_c': 1, ' AU10_c': 1, ' AU17_c': 1, ' AU25_c': 1, ' AU26_c': 1},
        'neutral': {}
    }

    # Assign missing AUs as 0
    full_AU_mapping = {au: mapping.get(emotion, {}).get(au, 0) for au in all_AUs_c}

    return full_AU_mapping



def process_openface_gaze(file_path):
    df = pd.read_csv(file_path)
    
    # Extract gaze angles
    gaze_x = df.iloc[:, 11]
    gaze_y = df.iloc[:, 12]
    
    return gaze_x, gaze_y

def compare_gaze_estimations(openface_gaze_x, openface_gaze_y, au_45, gaze_tracking_data):
    gaze_df = pd.DataFrame(gaze_tracking_data, columns=["Frame", "Gaze_Direction", "Eye_Openness"])
    
    mean_gaze_x = np.mean(openface_gaze_x)
    std_gaze_x = np.std(openface_gaze_x)

    threshold = std_gaze_x * 1

    # Convert OpenFace angles to direction labels
    openface_labels = []
    for i, x in enumerate(openface_gaze_x):
        if au_45[i] == 1:
            openface_labels.append("Blinking")
        elif x > mean_gaze_x + threshold:
            openface_labels.append("Looking Right")
        elif x < mean_gaze_x - threshold:
            openface_labels.append("Looking Left")
        else:
            openface_labels.append("Looking Center")
    print(mean_gaze_x + threshold)
    print(mean_gaze_x - threshold)
    # Compare gaze directions
    gaze_df = gaze_df.iloc[:-23]
    match_count = (gaze_df["Gaze_Direction"].values == openface_labels).sum()
    accuracy = match_count / len(gaze_df)

    print(f"🔍 OpenFace vs. GazeTracking Accuracy: {accuracy * 100:.2f}%")
    
    mean_gaze_y = np.mean(openface_gaze_y)
    std_gaze_y = np.std(openface_gaze_y)

    threshold = std_gaze_y * 1

    # Convert OpenFace angles to direction labels
    openface_labels = []
    for i, y in enumerate(openface_gaze_y):
        if au_45[i] == 1:
            openface_labels.append("Blinking")
        elif y > mean_gaze_y + threshold:
            openface_labels.append("Looking Right")
        elif y < mean_gaze_y - threshold:
            openface_labels.append("Looking Left")
        else:
            openface_labels.append("Looking Center")

    # Compare gaze directions
    print(mean_gaze_y + threshold)
    print(mean_gaze_y - threshold)
    match_count = (gaze_df["Gaze_Direction"].values == openface_labels).sum()
    accuracy = match_count / len(gaze_df)

    print(f"🔍 OpenFace_right vs. GazeTracking Accuracy: {accuracy * 100:.2f}%")
