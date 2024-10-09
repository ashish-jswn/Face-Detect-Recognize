import face_recognition
import pandas as pd
import os
import pickle

# Load student data
students_df = pd.read_csv('student.csv')

# Create a dictionary to store face encodings
student_encodings = {}

# Iterate through each student in the DataFrame
for index, row in students_df.iterrows():
    roll_number = row['roll_number']  # Assuming there's a roll_number column
    folder_path = f"photos/{roll_number}"  # Folder for the student's photos
    
    # Ensure the folder exists
    if os.path.isdir(folder_path):
        # Load all images in the student's folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                # Store encodings for each image
                for face_encoding in face_encodings:
                    # Use the student's name as the key and store the encoding
                    student_encodings[row['name']] = {
                        'encoding': face_encoding
                    }
            except Exception as e:
                print(f"Could not process image {image_path}: {e}")

# Save the encodings to a file
with open('face_encodings.pickle', 'wb') as f:
    pickle.dump(student_encodings, f)

print("Face encodings have been saved to 'face_encodings.pickle'")
