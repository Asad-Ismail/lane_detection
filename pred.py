import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from tensorflow.keras.optimizers import Adam
from utils import dataset
import  argparse
from utils import dataset
from utils import display_sample,show_predictions,create_mask
from Models import unetModel
import os
import datetime

parser=argparse.ArgumentParser(description="Input Args")
parser.add_argument("--train_images",type=str,default="/media/asad/8800F79D00F79104/LD_original_data/imageTar/images-4/")
parser.add_argument("--train_labels",type=str,default="/media/asad/8800F79D00F79104/LD_original_data/labelTar/labels-4/")
args=parser.parse_args()

if __name__=="__main__":
    mydata=dataset(args.train_images,args.train_labels)
    datasets=mydata.load_dataset()
    # Test and plot data
    data=list(datasets["train"].take(1).as_numpy_iterator())
    print(len(data))
    sample_image,sample_label=data[0][0],data[0][1]
    #display_sample([sample_image,sample_label])
    # Model declration 
    model=unetModel()
    unet=model.get_unet()
    #print(unet.summary())
    #print(dir(unet))
    unet.load_weights("best_unet_lane.h5")
    #tf.keras.utils.plot_model(unet, show_shapes=True)   
    unet.compile(optimizer=Adam(learning_rate=0.0001), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])    
    #sample show predictions
    intial_pred=create_mask(unet.predict(sample_image))
    show_predictions(sample_image=sample_image,sample_label=sample_label,sample_pred=intial_pred)