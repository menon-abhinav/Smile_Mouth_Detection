/**
 * @file FaceLandmark.cpp
 * @brief Implements face landmark detection using a TFLite model, including loading images, running inference, and extracting landmarks.
 */

#include "FaceLandmark.hpp"
#include <iostream>
#include <iomanip>  

#define FACE_LANDMARKS 468

// Each connection is a pair (start, end); only the start point is used.
// The connections are used to draw lines between the landmarks.
static const std::vector<std::pair<int, int>> FACE_LANDMARK_CONNECTIONS = {
    // Lips.
    {61, 146}, {146, 91}, {91, 181}, {181, 84}, {84, 17}, {17, 314},
    {314, 405}, {405, 321}, {321, 375}, {375, 291}, {61, 185}, {185, 40},
    {40, 39}, {39, 37}, {37, 0}, {0, 267}, {267, 269},
    {269, 270}, {270, 409}, {409, 291}, {78, 95}, {95, 88}, {88, 178},
    {178, 87}, {87, 14}, {14, 317}, {317, 402}, {402, 318}, {318, 324},
    {324, 308}, {78, 191}, {191, 80}, {80, 81}, {81, 82}, {82, 13}, {13, 312},
    {312, 311}, {311, 310}, {310, 415}, {415, 308},
    // Left eye.
    {33, 7}, {7, 163}, {163, 144}, {144, 145}, {145, 153}, {153, 154},
    {154, 155}, {155, 133}, {33, 246}, {246, 161}, {161, 160}, {160, 159},
    {159, 158}, {158, 157}, {157, 173}, {173, 133},
    // Left eyebrow.
    {46, 53}, {53, 52}, {52, 65}, {65, 55}, {70, 63}, {63, 105}, {105, 66},
    {66, 107},
    // Right eye.
    {263, 249}, {249, 390}, {390, 373}, {373, 374}, {374, 380}, {380, 381},
    {381, 382}, {382, 362}, {263, 466}, {466, 388}, {388, 387}, {387, 386},
    {386, 385}, {385, 384}, {384, 398}, {398, 362},
    // Right eyebrow.
    {276, 283}, {283, 282}, {282, 295}, {295, 285}, {300, 293}, {293, 334},
    {334, 296}, {296, 336},
    // Face oval.
    {10, 338}, {338, 297}, {297, 332}, {332, 284}, {284, 251}, {251, 389},
    {389, 356}, {356, 454}, {454, 323}, {323, 361}, {361, 288}, {288, 397},
    {397, 365}, {365, 379}, {379, 378}, {378, 400}, {400, 377}, {377, 152},
    {152, 148}, {148, 176}, {176, 149}, {149, 150}, {150, 136}, {136, 172},
    {172, 58}, {58, 132}, {132, 93}, {93, 234}, {234, 127}, {127, 162},
    {162, 21}, {21, 54}, {54, 103}, {103, 67}, {67, 109}, {109, 10},
};

/**
 * @brief Checks if the given index is valid for face landmarks.
 * 
 * @param idx The index to check.
 * @return True if the index is valid, false otherwise.
 */
bool __isIndexValid(int idx) {
    if (idx < 0 || idx >= FACE_LANDMARKS) {
        std::cerr << "Index " << idx << " is out of range (" \
        << FACE_LANDMARKS << ")." << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Constructor for FaceLandmark, initializes the model.
 * 
 * @param modelPath Directory containing the TFLite model.
 */
my::FaceLandmark::FaceLandmark(std::string modelPath):
    FaceDetection(modelPath),
    m_landmarkModel(modelPath + std::string("/TensorFlowFacialLandmarksV1.tflite"))
    {}

/**
 * @brief Runs inference on the loaded image and processes the result.
 */
void my::FaceLandmark::runInference() {
    FaceDetection::runInference();
    auto roi = FaceDetection::getFaceRoi();
    if (roi.empty()) return;

    auto face = FaceDetection::cropFrame(roi);
    m_landmarkModel.loadImageToInput(face);
    m_landmarkModel.runInference();

}

/**
 * @brief Gets the face landmark at the specified index.
 * 
 * @param index The index of the landmark to retrieve.
 * @return The coordinates of the landmark as a cv::Point.
 */
cv::Point my::FaceLandmark::getFaceLandmarkAt(int index) const {
    if (__isIndexValid(index)) {
        auto roi = FaceDetection::getFaceRoi();

        float _x = m_landmarkModel.getOutputData()[index * 3];
        float _y = m_landmarkModel.getOutputData()[index * 3 + 1];

        int x = (int)(_x / m_landmarkModel.getInputShape()[2] * roi.width) + roi.x;
        int y = (int)(_y / m_landmarkModel.getInputShape()[1] * roi.height) + roi.y;

        return cv::Point(x,y);
    }
    return cv::Point();
}

/**
 * @brief Gets all face landmarks.
 * 
 * @return A vector of all face landmarks as cv::Point objects.
 */
std::vector<cv::Point> my::FaceLandmark::getAllFaceLandmarks() const {
    if (FaceDetection::getFaceRoi().empty())
        return std::vector<cv::Point>();

    std::vector<cv::Point> landmarks(FACE_LANDMARKS);
    for (int i = 0; i < FACE_LANDMARKS; ++i) {
        landmarks[i] = getFaceLandmarkAt(i);
    }
    return landmarks;
}

/**
 * @brief Loads the output data from the landmark model.
 * 
 * @param index The index of the output tensor.
 * @return A vector of output data.
 */
std::vector<float> my::FaceLandmark::loadOutput(int index) const {
    return m_landmarkModel.loadOutput();
}

/**
 * @brief Normalizes the landmarks from the raw data.
 * 
 * @param raw_data The raw landmark data.
 * @param num_landmarks The number of landmarks.
 * @param tensor_width The width of the input tensor.
 * @param tensor_height The height of the input tensor.
 * @return A vector of normalized landmarks as cv::Point3f objects.
 */
std::vector<cv::Point3f> my::FaceLandmark::normalizeLandmarks(const float* raw_data, int num_landmarks, int tensor_width, int tensor_height) {
    std::vector<cv::Point3f> normalized;
    normalized.reserve(num_landmarks);
    for (int i = 0; i < num_landmarks; ++i) {
        float x = raw_data[i * 3]     / static_cast<float>(tensor_width);
        float y = raw_data[i * 3 + 1] / static_cast<float>(tensor_height);
        float z = raw_data[i * 3 + 2] / static_cast<float>(tensor_width);
        normalized.push_back(cv::Point3f(x, y, z));
    }
    return normalized;
}



/**
 * @brief Creates "points to train" from normalized landmarks.
 * 
 * @param normalized_landmarks The normalized landmarks.
 * @param connections The connections between landmarks.
 * @return A vector of points to train.
 */
std::vector<std::pair<float, float>> my::FaceLandmark::createPointsToTrain(const std::vector<cv::Point3f>& normalized_landmarks,
    const std::vector<std::pair<int, int>>& connections) {
    std::vector<std::pair<float, float>> points;
    for (const auto &conn : connections) {
        int start = conn.first;  // only the start index is used (as in Python)
        if (start >= 0 && start < static_cast<int>(normalized_landmarks.size())) {
            points.push_back({ normalized_landmarks[start].x, normalized_landmarks[start].y });
        } 
        else {
        std::cerr << "Connection start index " << start << " out of range." << std::endl;
        }
    }
    return points;
}


/**
 * @brief Gets normalized landmarks from the current model output.
 * 
 * @return A vector of normalized landmarks as cv::Point3f objects.
 */
std::vector<cv::Point3f> my::FaceLandmark::getNormalizedLandmarks() const {
    const float* raw_data = m_landmarkModel.getOutputData();
    const std::vector<int>& inputShape = m_landmarkModel.getInputShape();  // Typically [1, height, width, channels]
    int tensor_height = inputShape[1];  // e.g., 224
    int tensor_width  = inputShape[2];  // e.g., 224
    return normalizeLandmarks(raw_data, FACE_LANDMARKS, tensor_width, tensor_height);
}

/**
 * @brief Gets the "points to train".
 * 
 * @return A vector of points to train.
 */
std::vector<std::pair<float, float>> my::FaceLandmark::getPointsToTrain() const {
    
    std::vector<cv::Point3f> norm_landmarks = getNormalizedLandmarks();
    return createPointsToTrain(norm_landmarks, FACE_LANDMARK_CONNECTIONS);
}