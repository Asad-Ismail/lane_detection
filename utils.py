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
from random import sample 

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
    print(len(display_list))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    _, ax = plt.subplots(display_list[0].shape[0], len(display_list), sharex=True)
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

def pick_save_n_dataset(images_path,labels_path,dst_images,dst_label,n=20000,target_size=(640,384)):
    images=os.listdir(images_path)
    images=sample(images,n)  
    for i,image in enumerate(images):
        print(f"Procsses {i} data")
        c_i=os.path.join(images_path,image)
        c_l=os.path.join(labels_path,image)
        img=cv2.resize(cv2.imread(c_i),target_size,interpolation=cv2.INTER_CUBIC)
        label=cv2.resize(cv2.imread(c_l),target_size,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(dst_images,image),img)
        cv2.imwrite(os.path.join(dst_label,image),label)

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
        print(f"Training size {self.train_size}")
        print(f"Validation size {self.val_size}")
        #Load datasets
        #dataset = dataset.interleave(dataset.map(self.load, num_parallel_calls=AUTOTUNE))
        dataset = dataset.map(self.load,num_parallel_calls=AUTOTUNE)
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
        train_dataset=train_dataset.repeat()
        # No augmentation for the validation dataset
        val_dataset = val_dataset.batch(b_sz)
        val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
        val_dataset=val_dataset.repeat()
        return {"train":train_dataset,"val":val_dataset}


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))



def evaluate_lite_model(interpreter_path,test_data):
  interpreter = tf.lite.Interpreter(model_path=str(interpreter_path))
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  output_shape = interpreter.get_output_details()[0]["shape"]
  # Resize input tensor to take 150 images batch size
  #input_shape=[150,100,100,3]
  #interpreter.resize_tensor_input(input_index,input_shape)
  #interpreter.resize_tensor_input(output_index,[150, 1, output_shape[1]])  
  interpreter.allocate_tensors()
  # Run predictions on every image in the "test" dataset.
  results= []
  #print(f"Total test images batches {len(test_data)}")
  for i,(test_image,labels) in enumerate(test_data):
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    #if i==len(test_data)-1:
    #    break
    #test_image = test_image.astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output(),axis=-1)
    labels= np.squeeze(labels,axis=-1)
    #print(labels.size)
    res=np.sum((digit==labels).astype(np.uint8))/labels.size
    results.append(res)
  
  average_acc=np.mean(results)
  
  return average_acc


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
    #check_and_move("/media/asad/adas_cv_2/train/images","/media/asad/adas_cv_2/train/lane_labels","/media/asad/8800F79D00F79104/lanes_data/images","/media/asad/8800F79D00F79104/lanes_data/labels")
    #label_list=os.listdir("/media/asad/adas_cv_2/train/lane_labels")
    #label_list=[os.path.join("/media/asad/adas_cv_2/train/lane_labels",path) for path in label_list]
    #sample_list=sample(label_list,200)
    #get_class_weights(sample_list)
    #pick_save_n_dataset("/media/asad/8800F79D00F79104/lanes_data/images","/media/asad/8800F79D00F79104/lanes_data/labels","/media/asad/8800F79D00F79104/lanes_data/20k_images",
    #"/media/asad/8800F79D00F79104/lanes_data/20k_labels")
