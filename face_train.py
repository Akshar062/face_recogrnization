# pylint:disable=no-member

import cv2 as cv
import os
import numpy as np

people = []
DIR = 'samples'

# Load Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Create LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Open the default camera (index 0)
cap = cv.VideoCapture(0)

def collect_samples(person_name, num_samples=50):
    samples_collected = 0
    person_dir = os.path.join(DIR, person_name)

    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    while samples_collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y+h, x:x+w]

            # Save the collected sample
            cv.imwrite(os.path.join(person_dir, f'{person_name}_{samples_collected}.jpg'), faces_roi)
            samples_collected += 1

            cv.putText(frame, f'Samples Collected: {samples_collected}', (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=1)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

        cv.imshow('Collecting Samples', frame)
        cv.waitKey(1)

    print(f'Samples collected for {person_name}')

def create_train():
    while True:
        person_name = input("Enter the person's name (or 'q' to finish): ")
        if person_name.lower() == 'q':
            break

        # Ask for the number of samples to collect
        sample_number = input("Enter the number of samples to collect: ")
        if sample_number.isdigit():
            sample_number = int(sample_number)
        else:
            sample_number = 50

        people.append(person_name)

        with open('people_list.txt', 'w') as file:
            for person in people:
                file.write(person + '\n')

        print('Model trained and saved')
        print(f"Collecting samples for {person_name}. Press 'q' when done.")
        collect_samples(person_name, sample_number)

    features = []
    labels = []

    for label, person_name in enumerate(people):
        for i in range(50):
            new_dir = os.path.join(DIR, person_name)
            img_path = os.path.join(new_dir, f'{person_name}_{i}.jpg')

            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

    print('Training done ---------------')

    features = np.array(features, dtype='object')
    labels = np.array(labels)

    # Train the Recognizer on the features list and the labels list
    face_recognizer.train(features, labels)

    face_recognizer.save('face_trained.yml')
    np.save('features.npy', features)
    np.save('labels.npy', labels)

    print('Model trained and saved')

def main():
    print("Welcome to the Face Recognition Trainer!")
    print("Press 'q' to exit the program at any time.")

    # Check if the trained model and data exist
    if not (os.path.exists('face_trained.yml') and os.path.exists('features.npy') and os.path.exists('labels.npy')):
        create_train()

    # Load the face recognizer model
    if os.path.exists('face_trained.yml'):
        face_recognizer.read('face_trained.yml')
    else:
        print("Error: Could not load face recognizer model.")
        exit()

    # Load the people list from the text file
    if os.path.exists('people_list.txt'):
        with open('people_list.txt', 'r') as file:
            people.extend(line.strip() for line in file)

    # Print the people list to check if it's loaded correctly
    print("People list:", people)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y+h, x:x+w]

            # Perform face recognition
            label, confidence = face_recognizer.predict(faces_roi)

            # Check if the label is within the valid range
            if 0 <= label < len(people):
                label_text = f'Label = {people[label]} with confidence {confidence:.2f}'
                cv.putText(frame, label_text, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=1)
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            else:
                print(f'Error: Invalid label {label} detected.')

        cv.imshow('Live Face Recognition', frame)

        # Introduce a slight delay to allow time for key events
        cv.waitKey(1)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            new_person_name = input("Enter the name of the new person: ")
            people.append(new_person_name)
            number_of_samples = input("Enter the number of samples to collect: ")
            if number_of_samples.isdigit():
                number_of_samples = int(number_of_samples)
            else:
                number_of_samples = 50
            print(f"Collecting samples for {new_person_name}. Press 'q' when done.")
            collect_samples(new_person_name, number_of_samples)
            # Save the updated people list to the text file after collecting samples
            with open('people_list.txt', 'w') as file:
                for person in people:
                    file.write(person + '\n')
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
