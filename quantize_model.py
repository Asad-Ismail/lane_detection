import tensorflow as tf
import tensorflow.keras as K
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Models import unetModel
import pathlib
from utils import evaluate_lite_model
from utils import dataset

parser = argparse.ArgumentParser(description='Input Arguments')
parser.add_argument("--weight_file",type=str,default="/media/asad/8800F79D00F79104/ld_weights/best_unet_lane_20k_default_ce.h5")
parser.add_argument("--test_images",type=str,default="/media/asad/8800F79D00F79104/lanes_data/20k_images/")
parser.add_argument("--test_labels",type=str,default="/media/asad/8800F79D00F79104/lanes_data/20k_labels/")
args=parser.parse_args()

def post_quantization(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("./tf-lite-models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"model.tflite"
    tflite_model_file.write_bytes(tflite_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    tflite_model_quant_file = tflite_models_dir/"model_quant.tflite"
    tflite_model_quant_file.write_bytes(tflite_quant_model)
    print("Converted model to tflite!!!")
    return str(tflite_model_file),str(tflite_model_quant_file)


if __name__=="__main__":
    model=unetModel()
    unet=model.get_unet()
    unet.trainable=True
    local_weights=args.weight_file
    unet.load_weights(local_weights)
    tflite_model,tflite_quant_model=post_quantization(unet)
    print("Starting Evaluation of converted models ...")
    lane_data=dataset(args.test_images,args.test_labels)
    datasets=lane_data.load_dataset(b_sz=1)
    #lite_model_path,quantized_model_path=post_quantization(unet)
    print("."*20+"Evaluating Lite model"+"."*20)
    lite_model_acc=evaluate_lite_model(tflite_model,datasets['val'])
    print(f"Lite model has accuracy: {lite_model_acc}")
    print("."*20+"Evaluating Quantized Lite model"+"."*20)
    lite_model_acc=evaluate_lite_model(quantized_model_path,test_data)
    print(f"Quantized Lite model has accuracy: {lite_model_acc}")
