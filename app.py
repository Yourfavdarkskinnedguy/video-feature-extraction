import cv2 
import numpy as np
import os 
import math
import pytesseract
import re
from ultralytics import YOLO



#Input video filepath here
file_path= "/Users/daniel/Desktop/video feature extraction/video-feature-extraction/205614eacdcc431f85e859d97928bfc3.MOV"

#Input Yolo filepath here
model= YOLO('/Users/daniel/Desktop/video feature extraction/video-feature-extraction/yolov8n.pt')


previous_value = None
cut_count=0
mean_threshold= 1.9
text_frames = 0
total_frames = 0

persons_in_frame = 0
objects_in_frame = 0




def mean_calculator(previous_frame, current_frame):
    global cut_count
    
    previous_frame_mean= np.mean(previous_frame)
    current_frame_mean= np.mean(current_frame)

    mean_avg= current_frame_mean - previous_frame_mean
    mean_avg= abs(mean_avg)
    

    if mean_avg> mean_threshold:
        cut_count += 1

    return cut_count, mean_avg, previous_frame_mean, current_frame_mean

def quantify_average_motion(prev_frame, next_frame):
    # Convert frames to gray
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract the x and y components of the flow vectors
    dx = flow[..., 0]
    dy = flow[..., 1]

    #This is to Calculate the mean of the x and y components across all pixels
    mean_dx = np.mean(dx)
    mean_dy = np.mean(dy)

    #did this to Calculate the magnitude of each flow vector
    magnitude = np.sqrt(dx**2 + dy**2)

    #To Calculate the average magnitude of motion
    average_magnitude = np.mean(magnitude)

    return (mean_dx, mean_dy), average_magnitude


def text_present(processed_frame):
    text = pytesseract.image_to_string(processed_frame)
    return bool(re.search(r"[A-Za-z0-9]", text))




cap = cv2.VideoCapture(file_path)
overall_avg_motion=[]

while True:
    ret, current_value = cap.read()  # Read a full frame
    if not ret:
        break  # End of video

    if previous_value is not None:
        total_frames += 1
        if text_present(current_value):
            text_frames += 1
        text_present_ratio = text_frames / total_frames

        #YOLO
        results = model(current_value, verbose=False)

        for r in results:
            for cls in r.boxes.cls:
                class_name = model.names[int(cls)]
                if class_name == "person":

                    persons_in_frame += 1
                else:
                    objects_in_frame += 1
            

        cut_count, mean_avg, previous_frame_mean, current_frame_mean = mean_calculator(previous_value, current_value)


        (mean_dx, mean_dy), avg_magnitude = quantify_average_motion(previous_value, current_value)


        overall_avg_motion.append(avg_magnitude)

    # Update previous frame
    previous_value = current_value

cap.release()

mean_overall_avg_motion= np.mean(overall_avg_motion)
print(f'persons_in_frame: {persons_in_frame}')
print(f'objects_in_frame: {objects_in_frame}')

if objects_in_frame > 0:
    person_object_ratio = persons_in_frame / objects_in_frame

else:
    person_object_ratio = f'{persons_in_frame} : {objects_in_frame}'
        

total_output={
              'hard cut': cut_count,
              'average motion': mean_overall_avg_motion,
              'text_present_ratio': text_present_ratio,
              'person_object_ratio': person_object_ratio
          }


print(f'total_output: {total_output}')

    


