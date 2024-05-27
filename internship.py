import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow
import streamlit as st

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO


# Streamlit app title
st.title('Dental X-Ray Image Tooth Detection')

# Upload an image
uploaded_file = st.file_uploader("Choose a dental image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Display the original image
    st.subheader('Original Image')
    st.image(image, channels="BGR")

    # Perform YOLO object detection
    model = YOLO('regionOfIntrestModel.pt')
    results = model(image)

    if len(results) > 0:
        bounding_boxes = results[0].boxes.xyxy[0].cpu().numpy()
        x_min, y_min, x_max, y_max = map(int, bounding_boxes)

        # Draw bounding box on the original image
        cv2.rectangle(image, (x_min, y_max), (x_max, y_min), (0, 255, 0), 2)

        # Extract the cropped region from the original image
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        equalized_image = cv2.equalizeHist(gray_image)
        
        


        # Display the cropped image
        st.subheader('Pre-Processed image')
        st.image(equalized_image, channels="GRAY")

        # Save the cropped image as 'roi.jpg'
        cv2.imwrite('roi.jpg', cropped_image)

        # Perform Roboflow predictions
        rf = Roboflow(api_key="UBpLhl1PuRf0OwShbcwK")
        project = rf.workspace().project("individual-teeth-labelling-wa1cf")
        model = project.version(2).model
        image = 'roi.jpg'
        predictions = model.predict(image, confidence=40).json()
        model.predict(image, confidence=40).save("prediction2.jpg")


        #  list to store the extracted details
        extracted_details = []

        # Iterate through the 'predictions' in the JSON data
        for prediction in predictions['predictions']:
            details = {}

            # Extract x, y, width, and height
            if 'x' in prediction:
                details['x'] = prediction['x']
            if 'y' in prediction:
                details['y'] = prediction['y']
            if 'width' in prediction:
                details['width'] = prediction['width']
            if 'height' in prediction:
                details['height'] = prediction['height']

            # Add class and other details
            details['class'] = prediction['class']
            details['confidence'] = prediction['confidence']
            details['points'] = prediction['points']
            details['image_path'] = prediction['image_path']

            extracted_details.append(details)
        sorted_details = sorted(extracted_details, key=lambda x: x['class'])
        threshold = 5
        indexes_to_delete = []

        for i, detail in enumerate(sorted_details):
            label = detail['class']
            x = detail['x']
            y = detail['y']
            similar_entry_index = None
            for j in range(i + 1, len(sorted_details)):
                existing_detail = sorted_details[j]
                existing_x = existing_detail['x']
                existing_y = existing_detail['y']
                if abs(x - existing_x) <= threshold and abs(y - existing_y) <= threshold:
                    similar_entry_index = j
                    break
            if similar_entry_index is not None:
                indexes_to_delete.append(similar_entry_index)
        for i in reversed(indexes_to_delete):
            del sorted_details[i]
            
        sorted_details.sort(key=lambda x: x['x'])
        
        num_objects = len(sorted_details)

   
        label_mappings = {
            14: [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37],
            16: [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
        }

        new_labels = label_mappings.get(num_objects, [])

        for i, detail in enumerate(sorted_details):
            if i < len(new_labels):
                detail['class'] = new_labels[i]
            
        image_path = 'roi.jpg'
        image = cv2.imread(image_path)

        # Define class colors
        class_colors = {
            '31': (255, 0, 0),    # Red
            '32': (0, 255, 0),    # Green
            '33': (0, 0, 255),    # Blue
            '34': (255, 255, 0),  # Yellow
            '35': (0, 255, 255),  # Cyan
            '36': (255, 0, 255),  # Magenta
            '37': (128, 0, 0),    # Maroon
            '38': (0, 128, 0),    # Green 
            '41': (255, 0, 0),    # Red
            '42': (0, 255, 0),    # Green
            '43': (0, 0, 255),    # Blue
            '44': (255, 255, 0),  # Yellow
            '45': (0, 255, 255),  # Cyan
            '46': (255, 0, 255),  # Magenta
            '47': (128, 0, 0),    # Maroon
            '48': (0, 128, 0),    # Green 
        }

       
        for detail in sorted_details:
            x = int(detail['x'])
            y = int(detail['y'])
            width = int(detail['width'])
            height = int(detail['height'])
            class_label = str(detail['class']) 
            points = detail['points']

        
            if class_label in class_colors and '31' <= class_label <= '48':
                
                contour_points = np.array([(point['x'], point['y']) for point in points], np.int32)
                contour_points = contour_points.reshape((-1, 1, 2))
                cv2.drawContours(image, [contour_points], 0, (0,255,0), 5)

               
                label_position = (x, y)
               
        cv2.imwrite("labelled_image.jpg", image)


       
        st.subheader('Labelled Image')
        st.image("labelled_image.jpg", channels="GRAY")

            # Determine the length of extracted details
        length_extracted_details = len(sorted_details)

        # Print the length of extracted details
        st.subheader(f"Number of Tooths detected: {length_extracted_details}")

    else:
        st.warning("No objects detected in the image.")
else:
    st.info("Upload an image to get started.")