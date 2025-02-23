/**
 * @file demo.cpp
 * @brief Main application for smile and mouth detection using TFLite models.
 */

#include "FaceLandmark.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <string>
#include <cstdlib>
#include <tensorflow/lite/c/c_api_types.h>
#include <algorithm> 


// Forward declarations.
void processImage(const cv::Mat& image, my::FaceLandmark& faceLandmarker,
                  tflite::Interpreter* interpreter, float threshold,
                  const std::string& classifier, int landmarkCount);
void runWebcam(my::FaceLandmark& faceLandmarker, tflite::Interpreter* interpreter,
               float threshold, const std::string& classifier, int landmarkCount);
cv::Mat cropMouthRegion(const cv::Mat &image, const std::vector<std::pair<float, float>> &landmarks);



int main(int argc, char* argv[]) {
    
    // Default values for the parameters
    std::string imagePath = "../SmileMouth-PassportPhotos/2cd178fa9fad4f978e60cb797c169de7.jpg";  
    std::string classifier = "M";                 
    int landmarkCount = 40;                    
    float threshold = 0.1f;                      
    bool useWebcam = false;                        
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-I") {
            if (i + 1 < argc && argv[i+1][0] != '-') {
                imagePath = argv[++i];
            }
            
        } else if (arg == "-W") {
            useWebcam = true;
        } else if (arg == "-C") {
            if (i + 1 < argc && argv[i+1][0] != '-') {
                classifier = argv[++i];
            }
        } else if (arg == "-L") {
            if (i + 1 < argc && argv[i+1][0] != '-') {
                landmarkCount = std::stoi(argv[++i]);
            }
        } else if (arg == "-T") {
            if (i + 1 < argc && argv[i+1][0] != '-') {
                threshold = std::stof(argv[++i]);
            }
        } else {
            std::cerr << "Invalid argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0]
                      << " [ -C classifier ] [ -L landmark_count ] [ -T threshold ] [ -W | -I image_path ]" << std::endl;
            return 1;
        }
    }

    std::string modelPath;
    if (classifier == "M" || classifier == "m") {
        if (landmarkCount == 40) {
            modelPath = "../models/best_model_cropped_mouth_40_lm_attention.tflite";
        } else {
            modelPath = "../models/best_model_cropped_mouth_124_lm_attention.tflite";
        }
    } else { 
        if (landmarkCount == 40) {
            modelPath = "../models/best_model_whole_face_40_lm_attention.tflite";
        } else {
            modelPath = "../models/best_model_whole_face_124_lm_attention.tflite";
        }
    }

    std::cout << "Classifier type: " << classifier << std::endl;
    std::cout << "Landmark count: " << landmarkCount << std::endl;
    std::cout << "Threshold: " << threshold << std::endl;
    std::cout << "Model Path: " << modelPath << std::endl;

    // Create and Load the FaceLandmark object and the classifier TFlite model.
    my::FaceLandmark faceLandmarker("../models");

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model) {
        std::cerr << "Failed to load the TFLite model from: " << modelPath << std::endl;
        return 1;
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter." << std::endl;
        return 1;
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return 1;
    }

    if (useWebcam) {
        runWebcam(faceLandmarker, interpreter.get(), threshold, classifier, landmarkCount);
    } else {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Cannot read image: " << imagePath << std::endl;
            return 1;
        }
        processImage(image, faceLandmarker, interpreter.get(), threshold, classifier, landmarkCount);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

/**
 * @brief Processes a single image for smile and mouth detection.
 * 
 * @param image The input image to process.
 * @param faceLandmarker Reference to the FaceLandmark object for face landmark detection.
 * @param interpreter Pointer to the TFLite interpreter for running the classifier model.
 * @param threshold Threshold value for classifier decision.
 * @param classifier Type of classifier to use ("M" for mouth, others for smile).
 * @param landmarkCount Number of landmarks to use for the classifier.
 */
void processImage(const cv::Mat& image, my::FaceLandmark& faceLandmarker,
                  tflite::Interpreter* interpreter, float threshold,
                  const std::string& classifier, int landmarkCount) {
    cv::Mat rframe = image.clone();
    cv::flip(rframe, rframe, 1);

    // Run face landmark detection.
    faceLandmarker.loadImageToInput(rframe);
    faceLandmarker.runInference();

    // Get the ROI and crop the full face.
    cv::Rect roi = faceLandmarker.getFaceRoi();
    if (roi.empty()) {
        std::cerr << "No face detected!" << std::endl;
        return;
    }
    cv::Mat cropped_face = faceLandmarker.cropFrame(roi);
    cv::Mat origFace = cropped_face.clone();

    // ---------------------------------------------------------------------
    // Prepare image input for the classifier.
    cv::Mat imageInput;
    if (classifier == "M" || classifier == "m") {
        // For mouth classifier, first extract the mouth region.
        std::vector<std::pair<float, float>> pts = faceLandmarker.getPointsToTrain();
        std::vector<std::pair<float, float>> mouthLandmarks;
        if (pts.size() >= 40) {
            mouthLandmarks.assign(pts.begin(), pts.begin() + 40);
        } else {
            mouthLandmarks = pts;
        }
        cv::Mat mouthRegion = cropMouthRegion(origFace, mouthLandmarks);
        cv::resize(mouthRegion, imageInput, cv::Size(64, 64));
    } else {
        cv::resize(cropped_face, imageInput, cv::Size(224, 224));
    }

    imageInput.convertTo(imageInput, CV_32FC3, 1.0 / 255.0);
    if (!imageInput.isContinuous()) {
        imageInput = imageInput.clone();
    }
    float* imageInputTensor = interpreter->typed_input_tensor<float>(0);
    size_t numImageElements = (classifier == "M" || classifier == "m") ? (64 * 64 * 3) : (224 * 224 * 3);
    memcpy(imageInputTensor, imageInput.data, numImageElements * sizeof(float));

    // ---------------------------------------------------------------------
    // Prepare landmark input for the classifier. Use normalized landmarks from faceLandmarker.
    std::vector<std::pair<float, float>> ptsToTrain = faceLandmarker.getPointsToTrain();
    std::vector<std::pair<float, float>> landmarkPts;
    if (landmarkCount == 40) {
        if (ptsToTrain.size() < 40) {
            landmarkPts = ptsToTrain;
        } else {
            landmarkPts = std::vector<std::pair<float, float>>(ptsToTrain.begin(), ptsToTrain.begin() + 40);
        }
    } else {
        if (ptsToTrain.size() > static_cast<size_t>(landmarkCount)) {
            ptsToTrain.resize(landmarkCount);
        }
        landmarkPts = ptsToTrain;
    }
    std::vector<float> landmarkInputData;
    landmarkInputData.reserve(landmarkPts.size() * 2);
    for (const auto& pt : landmarkPts) {
        landmarkInputData.push_back(pt.first);
        landmarkInputData.push_back(pt.second);
    }
    size_t expectedLandmarkElements = landmarkPts.size() * 2;
    if (landmarkInputData.size() != expectedLandmarkElements) {
        std::cerr << "Warning: Expected " << expectedLandmarkElements
                  << " landmark elements, but got " << landmarkInputData.size() << std::endl;
    }
    float* landmarkInputTensor = interpreter->typed_input_tensor<float>(1);
    memcpy(landmarkInputTensor, landmarkInputData.data(), expectedLandmarkElements * sizeof(float));

    // ---------------------------------------------------------------------
    // Run classifier inference.
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Error during inference" << std::endl;
        return;
    }
    // Get classifier output (shape: [1, 2]).
    float* output = interpreter->typed_output_tensor<float>(0);
    std::string resultText;
    if (classifier == "M" || classifier == "m") {
        resultText = (output[0] >= threshold) ? "Mouth Closed" : "Mouth Open";
    } else {
        resultText = (output[0] >= threshold) ? "No Smile" : "Smile";
    }
    resultText += " (score: " + std::to_string(output[0]) + ")";
    std::cout << resultText << std::endl;

    // Render raw landmarks on the original frame.
    auto raw_landmarks = faceLandmarker.getAllFaceLandmarks();
    for (const auto& landmark : raw_landmarks) {
        cv::circle(rframe, landmark, 2, cv::Scalar(0, 255, 0), -1);
    }
    cv::putText(rframe, resultText, cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 196, 255), 2);
    cv::imshow("Face Detector", rframe);
}


/**
 * @brief Captures video from the webcam and processes each frame for smile and mouth detection.
 * 
 * @param faceLandmarker Reference to the FaceLandmark object for face landmark detection.
 * @param interpreter Pointer to the TFLite interpreter for running the classifier model.
 * @param threshold Threshold value for classifier decision.
 * @param classifier Type of classifier to use ("M" for mouth, others for smile).
 * @param landmarkCount Number of landmarks to use for the classifier.
 */
void runWebcam(my::FaceLandmark& faceLandmarker, tflite::Interpreter* interpreter,
               float threshold, const std::string& classifier, int landmarkCount) {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open the camera." << std::endl;
        return;
    }
    while (true) {
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty())
            break;
        processImage(frame, faceLandmarker, interpreter, threshold, classifier, landmarkCount);
        if (cv::waitKey(1) == 27) // Exit on ESC key.
            break;
    }
    cap.release();
}

/**
 * @brief Crops the mouth region from the full face image given a set of normalized landmark points.
 * 
 * @param image The full face image.
 * @param landmarks A vector of normalized landmark points (values in [0,1]).
 * @return A cropped image of the mouth region.
 */
cv::Mat cropMouthRegion(const cv::Mat &image, const std::vector<std::pair<float, float>> &landmarks) {
    float min_x = 1.0f, max_x = 0.0f, min_y = 1.0f, max_y = 0.0f;
    for (const auto &pt : landmarks) {
        min_x = std::min(min_x, pt.first);
        max_x = std::max(max_x, pt.first);
        min_y = std::min(min_y, pt.second);
        max_y = std::max(max_y, pt.second);
    }
    int img_width = image.cols;
    int img_height = image.rows;
    int left = std::max(0, static_cast<int>(min_x * img_width));
    int top = std::max(0, static_cast<int>(min_y * img_height));
    int right = std::min(img_width, static_cast<int>(max_x * img_width));
    int bottom = std::min(img_height, static_cast<int>(max_y * img_height));
    cv::Rect cropRect(left, top, right - left, bottom - top);
    return image(cropRect).clone();
}

