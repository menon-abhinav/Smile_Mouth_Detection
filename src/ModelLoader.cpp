/**
 * @file FaceLandmark.hpp
 * @brief Defines the FaceLandmark class for using a Mediapipe face detector and landmark model, including loading images, running inference, and extracting landmarks.
 */

#include "ModelLoader.hpp"
#include <iostream>
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

#define INPUT_NORM_MEAN 127.5f
#define INPUT_NORM_STD  127.5f

/**
 * @brief Constructor for ModelLoader, initializes the model and interpreter.
 * 
 * @param modelPath Path to the TFLite model file.
 */
my::ModelLoader::ModelLoader(std::string modelPath) {
    loadModel(modelPath.c_str());
    buildInterpreter();
    allocateTensors();
    fillInputTensors();
    fillOutputTensors();

    m_inputLoads.resize(getNumberOfInputs(), false);
}

/**
 * @brief Gets the shape of the input tensor at the specified index.
 * 
 * @param index The index of the input tensor.
 * @return A vector representing the shape of the input tensor.
 */
std::vector<int> my::ModelLoader::getInputShape(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].dims;

    return std::vector<int>();
}

/**
 * @brief Gets the data of the input tensor at the specified index.
 * 
 * @param index The index of the input tensor.
 * @return A pointer to the data of the input tensor.
 */
float* my::ModelLoader::getInputData(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].data;

    return nullptr;
}

/**
 * @brief Gets the size of the input tensor at the specified index.
 * 
 * @param index The index of the input tensor.
 * @return The size of the input tensor in bytes.
 */
size_t my::ModelLoader::getInputSize(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].bytes;

    return 0;
}

/**
 * @brief Gets the number of input tensors.
 * 
 * @return The number of input tensors.
 */
int my::ModelLoader::getNumberOfInputs() const {
    return m_inputs.size();
}

/**
 * @brief Gets the shape of the output tensor at the specified index.
 * 
 * @param index The index of the output tensor.
 * @return A vector representing the shape of the output tensor.
 */
std::vector<int> my::ModelLoader::getOutputShape(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].dims;
        
    return std::vector<int>();
}

/**
 * @brief Gets the data of the output tensor at the specified index.
 * 
 * @param index The index of the output tensor.
 * @return A pointer to the data of the output tensor.
 */
float* my::ModelLoader::getOutputData(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].data;

    return nullptr;
}

/**
 * @brief Gets the size of the output tensor at the specified index.
 * 
 * @param index The index of the output tensor.
 * @return The size of the output tensor in bytes.
 */
size_t my::ModelLoader::getOutputSize(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].bytes;

    return 0;
}

/**
 * @brief Gets the number of output tensors.
 * 
 * @return The number of output tensors.
 */
int my::ModelLoader::getNumberOfOutputs() const {
    return m_outputs.size();
}

/**
 * @brief Loads an image into the input tensor at the specified index.
 * 
 * @param inputImage The input image to load.
 * @param idx The index of the input tensor.
 */
void my::ModelLoader::loadImageToInput(const cv::Mat& inputImage, int idx) {
    if (isIndexValid(idx, 'i')) {
        cv::Mat resizedImage = preprocessImage(inputImage, idx); // Need optimize
        loadBytesToInput(resizedImage.data, idx);
    }
}

/**
 * @brief Loads raw bytes into the input tensor at the specified index.
 * 
 * @param data The raw bytes to load.
 * @param idx The index of the input tensor.
 */
void my::ModelLoader::loadBytesToInput(const void* data, int idx) {
    if (isIndexValid(idx, 'i')) {
        memcpy(m_inputs[idx].data, data, m_inputs[idx].bytes);
        m_inputLoads[idx] = true;
    }
}

/**
 * @brief Runs inference on the loaded input tensors.
 */
void my::ModelLoader::runInference() {
    inputChecker();
    m_interpreter->Invoke(); // Tflite inference
}

/**
 * @brief Loads the output data from the specified output tensor.
 * 
 * @param index The index of the output tensor.
 * @return A vector of output data.
 */
std::vector<float> my::ModelLoader::loadOutput(int index) const {
    if (isIndexValid(index, 'o')) {
        int sizeInByte = m_outputs[index].bytes;
        int sizeInFloat = sizeInByte / sizeof(float);

        std::vector<float> inference(sizeInFloat);
        memcpy(&(inference[0]), m_outputs[index].data, sizeInByte);
        
        return inference;
    }
    return std::vector<float>();
}


//-------------------Private methods start here-------------------

/**
 * @brief Loads the TFLite model from the specified file path.
 * 
 * @param modelPath The path to the TFLite model file.
 */
void my::ModelLoader::loadModel(const char* modelPath) {
    m_model = tflite::FlatBufferModel::BuildFromFile(modelPath);
    if (m_model == nullptr) {
        std::cerr << "Fail to build FlatBufferModel from file: " << modelPath << std::endl;
        std::exit(1);
    }  
}

/**
 * @brief Builds the TFLite interpreter with the specified number of threads.
 * 
 * @param numThreads The number of threads to use for the interpreter.
 */
void my::ModelLoader::buildInterpreter(int numThreads) {
    tflite::ops::builtin::BuiltinOpResolver resolver;

    if (tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
        std::exit(1);
    }
    m_interpreter->SetNumThreads(numThreads);
}

/**
 * @brief Allocates tensors for the TFLite interpreter.
 */
void my::ModelLoader::allocateTensors() {
    if (m_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        std::exit(1);
    }
}

/**
 * @brief Fills the input tensor information.
 */
void my::ModelLoader::fillInputTensors() {
    for (auto input: m_interpreter->inputs()) {
        TfLiteTensor* inputTensor =  m_interpreter->tensor(input);
        TfLiteIntArray* dims =  inputTensor->dims;

        m_inputs.push_back({
            inputTensor->data.f,
            inputTensor->bytes,
            dims->data,
            dims->size
        });
    }
}

/**
 * @brief Fills the output tensor information.
 */
void my::ModelLoader::fillOutputTensors() {
    for (auto output: m_interpreter->outputs()) {
        TfLiteTensor* outputTensor =  m_interpreter->tensor(output);
        TfLiteIntArray* dims =  outputTensor->dims;

        m_outputs.push_back({
            outputTensor->data.f,
            outputTensor->bytes,
            dims->data,
            dims->size
        });
    }
}

/**
 * @brief Checks if the specified index is valid for input or output tensors.
 * 
 * @param idx The index to check.
 * @param c The type of tensor ('i' for input, 'o' for output).
 * @return True if the index is valid, false otherwise.
 */
bool my::ModelLoader::isIndexValid(int idx, const char c) const {
    int size = 0;
    if (c == 'i')
        size = m_inputs.size();
    else if (c == 'o')
        size = m_outputs.size();
    else 
        return false;

    if (idx < 0 || idx >= size) {
        std::cerr << "Index " << idx << " is out of range (" \
        << size << ")." << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Checks if all input tensors have been loaded.
 * 
 * @return True if all input tensors have been loaded, false otherwise.
 */
bool my::ModelLoader::isAllInputsLoaded() const {
    return (
        std::find(m_inputLoads.begin(), m_inputLoads.end(), false)
     == m_inputLoads.end()); 
}

/**
 * @brief Checks if all input tensors have been loaded and resets the input load flags.
 */
void my::ModelLoader::inputChecker() {
    if (isAllInputsLoaded() == false) {
        std::cerr << "Input ";
        for (int i = 0; i < m_inputLoads.size(); ++i) {
            if (m_inputLoads[i] == false) {
                std::cerr << i << " ";
            }
        }
        std::cerr << "haven't been loaded." << std::endl;
        std::exit(1);
    }
    std::fill(m_inputLoads.begin(), m_inputLoads.end(), false);
}

/**
 * @brief Preprocesses the input image for the model.
 * 
 * @param in The input image.
 * @param idx The index of the input tensor.
 * @return The preprocessed image.
 */
cv::Mat my::ModelLoader::preprocessImage(const cv::Mat& in, int idx) const {
    auto out = convertToRGB(in);

    std::vector<int> inputShape = getInputShape(idx);
    int H = inputShape[1];
    int W = inputShape[2]; 

    cv::Size wantedSize = cv::Size(W, H);
    cv::resize(out, out, wantedSize);

    /*
    Equivalent to (out - mean)/ std
    */
    out.convertTo(out, CV_32FC3, 1 / INPUT_NORM_STD, -INPUT_NORM_MEAN / INPUT_NORM_STD);
    return out;
}

/**
 * @brief Converts the input image to RGB format.
 * 
 * @param in The input image.
 * @return The converted RGB image.
 */
cv::Mat my::ModelLoader::convertToRGB(const cv::Mat& in) const {
    cv::Mat out;
    int type = in.type();

    if (type == CV_8UC3) {
        cv::cvtColor(in, out, cv::COLOR_BGR2RGB);
    }
    else if (type == CV_8UC4) {
        cv::cvtColor(in, out, cv::COLOR_BGRA2RGB);
    }
    else {
        std::cerr << "Image of type " << type << " not supported" << std::endl;
        std::exit(1);
    }
    return out;
}