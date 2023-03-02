# 🚗 Ego and Side Lane Detection for ADAS Applications (Tensorflow 2.x) 🚘
## [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Asad-Ismail/lane_detection/issues) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAsad-Ismail%2Flane_detection&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## 🌟 Features
1. Segement and classify ego, left and right lane lines. The output can be visualized as shown below:
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/109520292-5b520e80-7aac-11eb-982d-0ff7c8d0ab9e.gif", alt="animated" />
  </p>
2. Apply pruning, clustering, and quantization to miniaturize the model, making it embedded system-ready.
3. C++ inference to use the resulting miniature model.

### Training dataset:
* The training data is TU simple lane detection dataset https://github.com/TuSimple/tusimple-benchmark
* Dataset is preprocessed to annotate the lanes in 4 categories left ego (label=2), right ego (label=1), right lane (label=3) and left lane line (label=4)
* Images are 1280x720 RGB and labels are 1280x720 gray scale images
* Image augmentations like rotation, flipping, saturation, brightness and contrast changes are applied randomly
* Data pipeline is made efficent using data interleaving and prefetch 
### Model:
* MobileNetV2 is used as the backbone network and then transposed convolutions are applied for upsampling with UNET like feature concatenation
* ResNets or EfficentNet can also be used as backbone for better performance
## Training and Prediction
1) Install requiremnets using "pip install -r requirements.txt"
2) run training using "python train.py --train_images [path to train images] --train_labels [path to train labels]"
3) Pred on image or video using pred.py or pred_video.py(also writes the blended video)
5) Prune cluster and quantize model weights and activations
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/109626664-19bf7300-7b41-11eb-8367-de783d1af713.png" alt="pruning",img width="550" />
  </p>
    <p align="center">
  
5) ~11x reduction of model size by applying above pipeline from 24MB to 2.2MB
<p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/109627067-85a1db80-7b41-11eb-96f7-107d4ae99224.gif"  alt="animated" />
  </p>

Pretrained weights avaiable at https://drive.google.com/drive/folders/1EhQ-8UoFv4rvMqe2mrJ4HFzZATd_Ee8c?usp=sharing
