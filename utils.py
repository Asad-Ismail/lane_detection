from glob import glob
import argparse
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img
import tensorflow_addons as tfa
import os
import matplotlib.pyplot as plt
import numpy as np
import mpldatacursor
import cv2
import shutil
from sklearn.utils import class_weight

# Defining some hyper parameters
IMG_SIZE=(384,640)


def flip(x: tf.Tensor,y: tf.Tensor) -> tf.Tensor:
    if (tf.random.uniform(()) > 0.5):
        x = tf.image.flip_left_right(x)
        #flip the labels 
        y = tf.image.flip_left_right(y)
        #Since the labels of lanes are realtive to ego vehichle we need to reverese the lanes labels also during hoizontal flipping
        y*=8
        y=tf.where(y==8,np.dtype('uint8').type(2),y)
        y=tf.where(y==16,np.dtype('uint8').type(1),y)
        y=tf.where(y==24,np.dtype('uint8').type(4),y)
        y=tf.where(y==32,np.dtype('uint8').type(3),y)
    return x,y

def color(x: tf.Tensor,y:tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x,y


def rotate(x: tf.Tensor,y:tf.Tensor) -> tuple:
    angle=tf.random.uniform((), minval=-5, maxval=5)
    x=tfa.image.rotate(x,angle*3.14/180)
    y=tfa.image.rotate(y,angle*3.14/180)
    return x,y

def sharpness(x: tf.Tensor,y:tf.Tensor) -> tuple:
    x*=255.0
    x = tf.cast(x,tf.uint8)
    sharp=tf.random.uniform((), minval=1, maxval=20)
    if (tf.random.uniform(())>0.01):
        x=tfa.image.sharpness(x,0.9)
    return x,y

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    #plt.figure(figsize=(18, 18))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    _, ax = plt.subplots(display_list[0].shape[0], 3, sharex=True)
    fax = ax.ravel()
    for k in range(0,display_list[0].shape[0]):
        for i in range(len(display_list)):
            index=(k*len(display_list))+i
            pil_img=tf.keras.preprocessing.image.array_to_img(display_list[i][k])
            fax[index].imshow(pil_img)
            fax[index].set_axis_off() 
    for i in range(ax.shape[1]):
        ax[-1, i].set_xlabel(title[i])
    mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def process_img(img,IMG_SIZE=(384,640)):
    image = tf.image.resize(img, IMG_SIZE, method='bilinear',antialias=True)
    image=image[tf.newaxis,...]
        # Normalize input image
    image/=255.0
    return image


def check_and_move(input_imgaes,input_labels,dst_images,dst_labels):
    images_files=os.listdir(input_imgaes)
    label_files=os.listdir(input_labels)
    total_files=0
    for image in images_files:
        if image in label_files:
             total_files+=1
             shutil.copy(os.path.join(input_imgaes,image), os.path.join(dst_images,image))
             shutil.copy(os.path.join(input_labels,image), os.path.join(dst_labels,image))
             print(f"Copied {total_files}") 


def get_class_weights(images_list):
    av_weights=[]
    bg=0
    weights_dict={0:0,1:0,2:0,3:0,4:0}
    for image in images_list:
        img=cv2.imread(image)
        un_labels=np.unique(img)
        weights=class_weight.compute_class_weight('balanced',un_labels,img.ravel())
        for key,value in zip(un_labels,weights):
            weights_dict[key]+=value
    for k,v in weights_dict.items():
        av_weights.append(v/len(images_list))
    print(av_weights)
    return av_weights


def show_predictions(dataset=None,sample_image=None,sample_label=None,sample_pred=None,num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display_sample([sample_image, sample_label,sample_pred])

class dataset():
    def __init__(self,images_path,labels_path):
        self.images_path=images_path
        self.labels_path=labels_path
        self.SEED=0
        self.total_images = len(glob(images_path + "*.png"))
        print(f"The Training Dataset contains {self.total_images} images.")
        self.total_labels = len(glob(labels_path + "*.png"))
        print(f"The Training labels contains {self.total_labels} images.")
        assert self.total_images==self.total_labels,"The images and labels should be equal"


    def load(self,img_path:str)-> dict:
        """Load image"""
        filename=(tf.strings.split(img_path,"/")[-1])
        mask_path=tf.strings.join([self.labels_path,filename])
        #filename=img_path.split("/")[-1]
        #image=load_img(img_path,color_mode="RGB",target_size=IMG_SIZE,interpolation="bicubic")
        #label=load_img(img_path,color_mode="gray",target_size=IMG_SIZE,interpolation="nearest")
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        #image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.image.resize(image, IMG_SIZE, method='bilinear',antialias=True)
        # Normalize input image
        image/=255.0
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask= tf.image.convert_image_dtype(mask, tf.uint8)
        mask=tf.image.resize(mask, IMG_SIZE, method='nearest')
        return image,mask
        
    def load_dataset(self,split=0.05,b_sz=3):
        # Dataset settitngs
        AUTOTUNE=tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.list_files(self.images_path + "*.png", seed=self.SEED)
        self.val_size=int(self.total_images*split)
        self.train_size=self.total_images-self.val_size
        #Load datasets
        dataset = dataset.map(self.load)
        #Dataset splitting
        dataset.shuffle(buffer_size=1000,seed=self.SEED)
        val_dataset = dataset.take(self.val_size)
        train_dataset = dataset.skip(self.val_size)
        #Train dataset augmentation
        train_dataset = train_dataset.batch(b_sz)
        transforms=[flip,color,rotate]
        for t in transforms:
            train_dataset=train_dataset.map(lambda x,y:t(x,y),num_parallel_calls=AUTOTUNE)
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        # No augmentation for the validation dataset
        val_dataset = val_dataset.batch(b_sz)
        val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
        return {"train":train_dataset,"val":val_dataset}




if  __name__== "__main__":
    print(f"Enabled Eager Execution: {tf.executing_eagerly()}")
    #img_path="/media/asad/cv_data5/datasets/ADAS/LD_original_data/imageTar/images-4/"
    #label_path="/media/asad/cv_data5/datasets/ADAS/LD_original_data/labelTar/labels-4/"

    img_path="/home/asad/ld/images/"
    label_path="/home/asad/ld/labels/"
    mydata=dataset(img_path,label_path)
    dataset=mydata.load_dataset()
    # Test and plot data
    data=list(dataset["train"].take(1).as_numpy_iterator())
    sample_image,sample_label=data[0][0],data[0][1]
    
    #sample_image, sample_mask = image, mask
    display_sample([sample_image,sample_label])
