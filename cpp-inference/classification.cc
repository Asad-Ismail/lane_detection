#include <cstdio>
#include <bits/stdc++.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core/ocl.hpp>


using namespace cv;
using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }



void average_time(std::unique_ptr<tflite::Interpreter> &interpreter,int samples)
{
  std::vector<float> times;
  chrono::steady_clock::time_point Tbegin, Tend;
  for (int i=0;i<samples;++i)
  {
  Tbegin = chrono::steady_clock::now();
  interpreter->Invoke();
  Tend = chrono::steady_clock::now();
  int f;
  f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
  //cout<<"The processing time is: "<<f <<"ms"<<endl;
  times.push_back(f);
  }
  cout<<"The average processing time is: "<<std::accumulate(times.begin(),times.end(),0)/times.size()<<"ms"<<std::endl;

}

void read_labels(string f_path,vector<string> &label)
{

  std::cout<<"Reading Labels"<<std::endl;
  ifstream infile(f_path);
  if (!infile.is_open())
  {
    cerr<<"<Cannot Open Label"<<endl;
    exit(-1);
  }
  std::string line;
  while (std::getline(infile, line))
  {
    //cout<<line<<endl;
    label.push_back(line);
  }

}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: classification <tflite model> <labelfile> <imagename>\n");
    return 1;
  }
  const char* filename = argv[1];
  const char* labelfile = argv[2];
  const char* imagepath = argv[3];

  int model_height,model_width,model_channels;
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);
  vector<string> label_list;
  read_labels(labelfile,label_list);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  //For gpu tensors
  auto* delegate = TfLiteGpuDelegateV2Create(nullptr);
  if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

  // Allocate tensor buffers.
  //TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //printf("=== Pre-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  // Run inference
  //TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  //printf("\n\n=== Post-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());
  interpreter->SetAllowFp16PrecisionForFp32(true);
  interpreter->SetNumThreads(1);      //quad core

    // Get input dimension from the input tensor metadata
    // Assuming one input only
  int In;
  In = interpreter->inputs()[0];
  model_height   = interpreter->tensor(In)->dims->data[1];
  model_width    = interpreter->tensor(In)->dims->data[2];
  model_channels = interpreter->tensor(In)->dims->data[3];

  cout << "height   : "<< model_height << endl;
  cout << "width    : "<< model_width << endl;
  cout << "channels : "<< model_channels << endl;

    
  cv::Mat img;

  img=imread(imagepath);  //need to refresh frame before dnn class detection
    
    if (img.empty()) {
        cerr << "Can not load picture!" << endl;
        exit(-1);
    }
    
    // copy image to input as input tensor
    // Preprocess the input image to feed to the model
    cv:cvtColor(img,img,cv::COLOR_RGB2BGR);
    cv::resize(img, img, Size(model_width,model_height),INTER_CUBIC);
    img.convertTo(img, CV_32F);
    img/=255.0;
    memcpy(interpreter->typed_input_tensor<_Float32>(0), img.data, img.total() * img.elemSize());
    cout << "inputs: " << interpreter->inputs().size() << "\n";
    cout << "outputs: " << interpreter->outputs().size() << "\n";
    // Benchmark the timings of forward pass of model for n number of iterations
    average_time(interpreter,1000);

    interpreter->Invoke();      // run your model


    const float threshold = 0.001f;

    std::vector<std::pair<float, int>> top_results;

    int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    const int output_size = output_dims->data[output_dims->size - 1];
    cout << "output_size: " << output_size <<"\n";
    // Read output buffers
    // TODO(user): Insert getting data out code.
    // Note: The buffer of the output tensor with index `i` of type T can
    // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
    
    //std::vector <float> myout(output_size);
    //std::copy(interpreter->typed_output_tensor<float>(0), interpreter->typed_output_tensor<float>(0) + output_size,myout.begin());
    //for (int i=0;i<131;++i)
    //  cout<<"my out is" <<myout[i]<<endl;

    switch (interpreter->tensor(output)->type) {
        case kTfLiteFloat32:
            cout<<"Using TF Lite Float32"<<endl;
            tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                                                    5, threshold, &top_results, kTfLiteFloat32);
        break;
        case kTfLiteUInt8:
            cout<<"Using TF Lite Int8"<<endl;
            tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size,
                                                    5, threshold, &top_results, kTfLiteUInt8);
        break;
        default:
            cerr << "cannot handle output type " << interpreter->tensor(output)->type << endl;
            exit(-1);
  }

    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        cout<<"The image is: "<< label_list[index]<<" with score: "<<confidence<<endl;
    }


  return 0;
}
