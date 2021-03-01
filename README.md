# Ego and Side Lane Detection for ADAS applications
### Purpose of the repo is to provide complete pipeline for lane detection to be used for ADAS functionality like lane keeping and lane change
### Features:

1) Segement and classify left ego and left and right lane lines
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/109520292-5b520e80-7aac-11eb-982d-0ff7c8d0ab9e.gif" alt="animated" />
  </p>
2) Apply pruning, clustering and quantization to miniaturize the model and making it embedded system ready 
4) C++ inference to use the resulting miniature model.

### Training dataset:
* The training data is TU simple lane detection dataset https://github.com/TuSimple/tusimple-benchmark
* Dataset is preprocessed to annotate the lanes in 4 categories left ego (label=2), right ego (label=1), right lane (label=3) and left lane line (label=4)
* Images are 1280x720 RGB and labels are 1280x720 gray scale images
* Image augmentations like rotation, flipping, saturation, brightness and contrast changes are applied randomly
* Data pipeline is made efficent using data interleaving and prefetch 
### Model:
* MobileNetV2 is used as the backbone network and then transposed convolutions are applied for upsampling with UNET like feature concatenation
* ResNets or EfficentNet can also be used as backbone for better performance
