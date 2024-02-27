# Face Recognition Trainer

Welcome to the Face Recognition Trainer project! This Python application allows you to collect and train a face recognition model using OpenCV.

## Getting Started

### Prerequisites

- Python (3.6 and above)
- OpenCV (4.0 and above)

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/face-recognition-trainer.git
   cd face-recognition-trainer
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

# Usages

## Run the main script:
   ```bash
   python face_train.py
   ```

## Features

1. Collect face samples for multiple individuals.
2. Train a face recognition model using LBPH.
3. Real-time face recognition from a live camera feed.
4. Add new people and collect samples on the fly.


## File Structure

1. face_recognition_trainer.py: Main script for running the Face Recognition Trainer.
2. haar_face.xml: Haar Cascade file for face detection.
3. samples/: Directory for storing collected face samples.
4. face_trained.yml: Trained LBPH face recognizer model.
5. features.npy and labels.npy: Numpy arrays containing face recognition training data.
6. people_list.txt: Text file storing the names of recognized people.

# Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Your contributions are welcome!

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments

1. OpenCV community for providing the face recognition tools.
2. Contributors to the Haar Cascade project.
   
