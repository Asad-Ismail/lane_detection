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

#define TFLITE_MINIMAL_CHECK(x)                              
  if (!(x)) {                                                
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); 
    exit(1);                                                 
  }

  int model_height,model_width,model_channels;

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

struct RGB {
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

void read_labels(string f_path,vector<RGB> &label)
{

  std::cout<<"Reading Labels"<<std::endl;
  ifstream infile(f_path);
  if (!infile.is_open())
  {
    cerr<<"<Cannot Open Label"<<endl;
    exit(-1);
  }
  std::string line;
  string token;
  char split_with=' ';
  while (std::getline(infile, line))
  {
    stringstream ss(line);
    vector <int> temp;
    while(getline(ss , token , split_with))
    {
    temp.push_back(stoi(token));
    }
    label.push_back({(unsigned char)(temp[0]),(unsigned char)(temp[1]),(unsigned char)(temp[2])});
  }
  std::cout<<"Labels are :"<<endl;
  for (auto a: label)
    std::cout<<int(a.blue)<<" "<<int(a.green)<<" "<<int(a.red)<<" "<<endl;

}
// If gpu is avaiable replace this with cuda gpu since it takes significant time for semantic segmentaiton to loop through complete output
void argmax_cpu(RGB*& rgb,float*& data,vector<RGB> &label_list)
{
    int i,j,k,mi;
    float mx,v;
    for(i=0;i<model_height;i++)
    {
        for(j=0;j<model_width;j++)
        {
            for(mi=-1,mx=0.0,k=0;k<5;k++)
            {
                v = data[5*(i*model_width+j)+k];
                if(v>mx)
                {
                   mi=k;
                   mx=v; 
                }
            }
            rgb[j+i*model_width] = label_list[mi];
        }
    }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: segmentation <tflite model> <labelfile> <imagename>\n");
    return 1;
  }
  const char* filename = argv[1];
  const char* labelfile = argv[2];
  const char* imagepath = argv[3];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);
  vector<RGB> label_list;
  read_labels(labelfile,label_list);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);
  // For cpu tensor
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //For gpu tensors
  //auto* delegate = TfLiteGpuDelegateV2Create(nullptr);
  //if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
  // return false;
  interpreter->SetAllowFp16PrecisionForFp32(true);
  interpreter->SetNumThreads(4);      //quad core
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
  cv::Mat src;
  src=imread(imagepath);  //need to refresh frame before dnn class detection
  if (src.empty()) 
  {
        cerr << "Can not load picture!" << endl;
        exit(-1);
  }
  // copy image to input as input tensor
  // Preprocess the input image to feed to the model
  cv::Mat input=src.clone();
  cv:cvtColor(input,input,cv::COLOR_RGB2BGR);
  cv::resize(input, input, Size(model_width,model_height),INTER_CUBIC);
  input.convertTo(input, CV_32F);
  input/=255.0;
  memcpy(interpreter->typed_input_tensor<_Float32>(0), input.data, input.total() * input.elemSize());

  static Mat pred(model_height,model_width,CV_8UC3);
  interpreter->Invoke();      // run your model
  // get max object per pixel
  float *data = interpreter->tensor(interpreter->outputs()[0])->data.f;
  RGB * rgb = (RGB *)pred.data;
  argmax_cpu(rgb,data,label_list);
  //merge output into frame
  cv::resize(pred, pred, Size(src.cols,src.rows),INTER_NEAREST);
  cv::addWeighted(src, 0.5, pred, 0.5, 0.0, pred);
  // write output image
  cv::imwrite("out.png",pred);
  return 0;
}
