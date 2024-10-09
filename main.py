import face_recognition
import pandas as pd
import cv2
import numpy as np
import os
import pickle

# Load face encodings from the pickle file
with open('face_encodings.pickle', 'rb') as f:
    student_encodings = pickle.load(f)

def recognize_faces(input_image_path):
    # Load the input image
    input_image = face_recognition.load_image_file(input_image_path)
    
    # Detect faces in the input image
    face_locations = face_recognition.face_locations(input_image)
    face_encodings = face_recognition.face_encodings(input_image, face_locations)
    
    # Convert the image to BGR for OpenCV
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Initialize results
    results = []
    confidence_threshold = 0.6  # Set your confidence threshold

    # Loop over each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Find the best match for the detected face
        matches = face_recognition.compare_faces([student['encoding'] for student in student_encodings.values()], face_encoding)
        face_distances = face_recognition.face_distance([student['encoding'] for student in student_encodings.values()], face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # Check confidence
        if matches[best_match_index] and face_distances[best_match_index] < confidence_threshold:
            matched_name = list(student_encodings.keys())[best_match_index]
            
            # Add result to the list
            results.append((top, right, bottom, left, matched_name))
            
            # Draw a box around the face
            cv2.rectangle(input_image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with the student's name
            label = f"{matched_name}"
            cv2.putText(input_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Mark face as unknown
            results.append((top, right, bottom, left, "Unknown"))
            # Draw a box around the face with a red rectangle
            cv2.rectangle(input_image, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw label "Unknown"
            label = "Unknown"
            cv2.putText(input_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return input_image, results

def show_image(image):
    # Show the result
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    input_image_path = 'input.jpg'  # Path to the input image
    image, results = recognize_faces(input_image_path)
    show_image(image)
