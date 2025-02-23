# Smile and Mouth Detection using TFLite Models  

## Overview  
This project implements a **smile and mouth detection system** using **TensorFlow Lite** models. The program processes an image or a live webcam feed, detects facial landmarks, and classifies whether the subject is **smiling or not** and whether their **mouth is open or closed**.  

## Features  
- **Facial landmark detection** using a custom TFLite model.  
- **Smile and mouth classification** based on detected landmarks.  
- Supports **both static images and live webcam input**.  
- Configurable **thresholds, classifiers, and landmark points** for flexible testing.  

## Dependencies  
- OpenCV (`opencv-python`)  
- TensorFlow Lite (`tensorflow`)  
- C++ compiler supporting C++17  

## How to Run the Code

### Building the Project

1. Change the directory:
    ```sh
    cd src
    ```

2. Run CMake to configure the project:
    ```sh
    cmake .
    ```

3. Build the project:
    ```sh
    make
    ```

### Running the Application

The application can be run with various options to specify the input image, classifier type, landmark count, threshold, and whether to use the webcam.

#### Default Run

To run the application with default parameters:
```sh
./SmileMouthDetection
```

### Default Values

The application uses the following default values for its parameters:

- **Image Path**: `../SmileMouth-PassportPhotos/2cd178fa9fad4f978e60cb797c169de7.jpg`
- **Classifier**: `"M"` (Mouth classifier)
- **Landmark Count**: `40`
- **Threshold**: `0.1`
- **Use Webcam**: `false`

These default values can be overridden using command-line arguments as described below:

### Best Values

The best values for the classifiers were found to be:

#### Mouth Classifier

- **Threshold**: `0.1`
- **Number of Landmarks**: `40`

#### Smile Classifier

- **Threshold**: `0.75`
- **Number of Landmarks**: `124`

#### Command-Line Arguments

-I <image_path>: Specify the path to the input image.
-W: Use the webcam for real-time detection.
-C <classifier>: Specify the classifier type ("M" for mouth, others for smile).
-L <landmark_count>: Specify the number of landmarks to use for the classifier.
-T <threshold>: Specify the threshold value for classifier decision.

#### Examples

1. Run with a specific image:
    ```sh
    ./SmileMouthDetection -I <Image_Path>
    ```

2. Run with the webcam (Default Mouth Open / Closed Model):
    ```sh
    ./SmileMouthDetection -W
    ```

4. Run with a specific classifier and threshold:
    ```sh
    ./SmileMouthDetection -C M -T 0.5
    ```

#### Sample Run Examples

1. Run with a specific image:
    ```sh
    ./SmileMouthDetection -I "../SmileMouth-PassportPhotos/0b86598e3eea4ff7a5e115cec889b0f8.jpg"
    ```

2. Run with a specific classifier and threshold:
    ```sh
    ./SmileMouthDetection -C S -T 0.75 -I "../SmileMouth-PassportPhotos/0b86598e3eea4ff7a5e115cec889b0f8.jpg"
    ```

3. Run with a specific classifier, threshold, and landmark count:
    ```sh
    ./SmileMouthDetection -C M -T 0.1 -L 40 -I "../SmileMouth-PassportPhotos/0b86598e3eea4ff7a5e115cec889b0f8.jpg"
    ```

4. Run with a specific landmark count:
    ```sh
    ./SmileMouthDetection -L 124 -I "../SmileMouth-PassportPhotos/0b86598e3eea4ff7a5e115cec889b0f8.jpg"
    ```

5. Run with Webcam for Mouth Classifier on default Values:
    ```sh
    ./SmileMouthDetection -W
    ```

## Models Used

### Face Detection Model

- **Model Path**: `../models/face_detection_short.tflite`
- **Description**: This model detects faces in the input image and provides the region of interest (ROI) for further processing.

### Face Landmark Model

- **Model Path**: `../models/face_landmark.tflite`
- **Description**: This model detects 468 facial landmarks within the ROI provided by the face detection model.

### Classifier Models

- **Mouth Classifier**:
  - **Model Path**: `../models/best_model_cropped_mouth_40_lm_attention.tflite` or `../models/best_model_cropped_mouth_124_lm_attention.tflite`
  - **Description**: This model classifies the state of the mouth (open or closed) based on the detected landmarks.

- **Smile Classifier**:
  - **Model Path**: `../models/best_model_whole_face_40_lm_attention.tflite` or `../models/best_model_whole_face_124_lm_attention.tflite`
  - **Description**: This model classifies the presence of a smile based on the detected landmarks.

### How Models are Used

1. **Face Detection**: The face detection model is used to detect faces in the input image or video frame. The detected face's ROI is then used for further processing.

2. **Face Landmark Detection**: The face landmark model is used to detect 468 facial landmarks within the detected face's ROI.

3. **Classification**: Based on the specified classifier type, the appropriate classifier model is used to classify the state of the mouth (open or closed) or the presence of a smile. The classifier uses the detected landmarks as input.