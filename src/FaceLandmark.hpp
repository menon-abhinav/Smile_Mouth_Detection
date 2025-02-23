/**
 * @file FaceLandmark.hpp
 * @brief Defines the FaceLandmark class for using a Mediapipe face detector and landmark model, including loading images, running inference, and extracting landmarks.
 */

#ifndef FACELANDMARK_H
#define FACELANDMARK_H

#include "FaceDetection.hpp"

namespace my {

    /*
    A model wrapper to use Mediapipe Face Detector.
    It also includes the detection phase.  
    This class is non-copyable.
    */
    class FaceLandmark : public my::FaceDetection {
        public:
            /*
            Users MUST provide the FOLDER contain BOTH the face_detection_short.tflite 
            and face_landmark.tflite, 
            */
            FaceLandmark(std::string modelPath);
            virtual ~FaceLandmark() = default; 

            /*
            Override function from ModelLoader
            */
            virtual void runInference();

            /*
            Get a landmark from output (index must be in range 0-467)
            The position is relative to the input image at InputTensor(0)
            */
            virtual cv::Point getFaceLandmarkAt(int index) const;

            /*
            Get all landmarks from output.
            The positions is relative to the input image at InputTensor(0)
            */
            virtual std::vector<cv::Point> getAllFaceLandmarks() const;

            /*
            Get all landmarks from output, which is a vector of length 468 * 3 * 4 (although the first 468 * 3 are enough).
            (Note: index does not matter, it always load from OutputTensor(0))
            Each landmark is represented by x, y, z(depth), which are raw outputs from Mediapipe Face Landmark model.
            If you want to get relative position to input image, use getAllFaceLandmarks() or getFaceLandmarkAt()
            */
            virtual std::vector<float> loadOutput(int index = 0) const;

            // Declaration for our helper function to normalize raw landmarks.
            // This will take the raw landmark output and return normalized landmarks.
            static std::vector<cv::Point3f> normalizeLandmarks(const float* raw_data, int num_landmarks, int tensor_width, int tensor_height);

            // Each connection is a pair (start, end); only the start point is used.
            static std::vector<std::pair<float, float>> createPointsToTrain(const std::vector<cv::Point3f>& normalized_landmarks,
            const std::vector<std::pair<int, int>>& connections);

            // (Optional) Member functions that wrap the static helpers using the current model output.
            std::vector<cv::Point3f> getNormalizedLandmarks() const;
            std::vector<std::pair<float, float>> getPointsToTrain() const;


        private:
            my::ModelLoader m_landmarkModel;

    };
}

#endif // FACELANDMARK_H