#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "calibrator.h"
#include "alignface.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
#define VIS_THRESH 0.6

// stuff we know about the network and the input/output blobs
static const int INPUT_H = decodeplugin::INPUT_H;  // H, W must be able to  be divided by 32.
static const int INPUT_W = decodeplugin::INPUT_W;;
static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


// stuff we know about the network and the input/output blobs
static const int INPUT_FEATURE_H = 112;
static const int INPUT_FEATURE_W = 112;
static const int OUTPUT_FEATURE_SIZE = 512;
// const char* INPUT_FEATURE_BLOB_NAME = "data";
// const char* OUTPUT_FEATURE_BLOB_NAME = "prob";

static Logger gLogger;


void doInference(IExecutionContext& context, float* input, float* output, int batchSize, const int INPUT_H, const int INPUT_W, const int OUTPUT_SIZE) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}






int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./retina_mnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *det_trtModelStream{nullptr};
    char *fext_trtModelStream{nullptr};

    size_t size{0};
    std::ifstream file("../retina_mnet.engine", std::ios::binary);
    if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            det_trtModelStream = new char[size];
            assert(det_trtModelStream);
            file.read(det_trtModelStream, size);
            file.close();
    } else {
        std::cout<< "cannot read model! \n"; 
        return -1;
    }

    IRuntime* det_runtime = createInferRuntime(gLogger);
    assert(det_runtime != nullptr);
    ICudaEngine* det_engine = det_runtime->deserializeCudaEngine(det_trtModelStream, size);
    assert(det_engine != nullptr);
    IExecutionContext* det_context = det_engine->createExecutionContext();
    assert(det_context != nullptr);
    delete[] det_trtModelStream;


     std::ifstream fext_file("../arcface-r50.engine", std::ios::binary);
    if (fext_file.good()) {
            fext_file.seekg(0, fext_file.end);
            size = fext_file.tellg();
            fext_file.seekg(0, fext_file.beg);
            fext_trtModelStream = new char[size];
            assert(fext_trtModelStream);
            fext_file.read(fext_trtModelStream, size);
            fext_file.close();
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./arcface-r50 -d  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    IRuntime* fext_runtime = createInferRuntime(gLogger);
    assert(fext_runtime != nullptr);
    ICudaEngine* fext_engine = fext_runtime->deserializeCudaEngine(fext_trtModelStream, size);
    assert(fext_engine != nullptr);
    IExecutionContext* fext_context = fext_engine->createExecutionContext();
    assert(fext_context != nullptr);
    delete[] fext_trtModelStream;



    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;

    cv::Mat img = cv::imread("../worlds-largest-selfie.jpg");
    // preprocessing the image
    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);

    // in case checking input
    //cv::imwrite("preprocessed.jpg", pr_img);


    // For multi-batch, I feed the same image multiple times.
    // Filling the image in the right order.
    for (int b = 0; b < BATCH_SIZE; b++) {
        float *p_data = &data[b * 3 * INPUT_H * INPUT_W];

        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
            p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
            p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
        }
    }


    // Run face detection here
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    auto start = std::chrono::system_clock::now();
    doInference(*det_context, data, prob, BATCH_SIZE, INPUT_H, INPUT_W, OUTPUT_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;

    // postprocessing the faces
    for (int b = 0; b < BATCH_SIZE; b++) {
        std::vector<decodeplugin::Detection> res;
        nms(res, &prob[b * OUTPUT_SIZE]);
        std::cout << "number of detections -> " << prob[b * OUTPUT_SIZE] << std::endl;
        std::cout << " -> " << prob[b * OUTPUT_SIZE + 10] << std::endl;
        std::cout << "after nms -> " << res.size() << std::endl;
        
        cv::Mat tmp = img.clone();
        for (size_t j = 0; j < res.size(); j++) {
            if (res[j].class_confidence < VIS_THRESH) continue;
            cv::Rect r = get_rect_adapt_landmark(tmp, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark);
            cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
            for (int k = 0; k < 10; k += 2) {
                cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
            }
        }

        // in case to see the detections
        cv::imwrite(std::to_string(b) + "_result.jpg", tmp);

            // feading the detected images to the feature extraction
        // face alignmetn
        // std::vector<unsigned char> face;
        // cv::Mat input_face = alignface( cv::Mat src, int srcWidth, int srcHeight int dstWidth, int dstHeight, fp_t facelandmarks[10])

        cv::Mat input_face =  alignface(img, INPUT_W, INPUT_H, INPUT_FEATURE_W, INPUT_FEATURE_H,  res[0].landmark);

        // img buffer to opencv mat

        std::cout<< " aligned face size: " <<  input_face.cols << " <-=|=-> "  << input_face.rows << std::endl;


        // prepare input data ---------------------------
        static float fext_data[BATCH_SIZE * 3 * INPUT_FEATURE_H * INPUT_FEATURE_W];
        //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        //    data[i] = 1.0;
        static float fext_prob[BATCH_SIZE * OUTPUT_FEATURE_SIZE];


        for (int i = 0; i < INPUT_FEATURE_H * INPUT_FEATURE_W; i++) {
            fext_data[i] = ((float)input_face.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
            fext_data[i + INPUT_FEATURE_H * INPUT_FEATURE_W] = ((float)input_face.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
            fext_data[i + 2 * INPUT_FEATURE_H * INPUT_FEATURE_W] = ((float)input_face.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
        }

        auto start = std::chrono::system_clock::now();
        doInference(*fext_context, fext_data, fext_prob, BATCH_SIZE, INPUT_FEATURE_H, INPUT_FEATURE_W, OUTPUT_FEATURE_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::Mat out(512, 1, CV_32FC1, fext_prob);
        cv::Mat out_norm;
        cv::normalize(out, out_norm);

        std::cout<< "out: "<< out_norm << std::endl;

    }




















    // Destroy the engine
    det_context->destroy();
    det_engine->destroy();
    det_runtime->destroy();
    fext_context->destroy();
    fext_engine->destroy();
    fext_runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
