import tensorflow as tf
import  argparse
from utils import dataset
from utils import display_sample,show_predictions,create_mask,process_img
from Models import unetModel
import os
import datetime
import cv2
import numpy as np
from utils import check_and_move,get_class_weights,pick_save_n_dataset
from random import sample 

#check_and_move("/media/asad/adas_cv_2/train/images","/media/asad/adas_cv_2/train/lane_labels","/media/asad/8800F79D00F79104/lanes_data/images","/media/asad/8800F79D00F79104/lanes_data/labels")
#label_list=os.listdir("/media/asad/adas_cv_2/train/lane_labels")
#label_list=[os.path.join("/media/asad/adas_cv_2/train/lane_labels",path) for path in label_list]
#sample_list=sample(label_list,2)
#get_class_weights(sample_list)
pick_save_n_dataset("/media/asad/8800F79D00F79104/lanes_data/images","/media/asad/8800F79D00F79104/lanes_data/labels","/media/asad/8800F79D00F79104/lanes_data/20k_images",
"/media/asad/8800F79D00F79104/lanes_data/20k_labels")

parser=argparse.ArgumentParser(description="Input Args")
parser.add_argument("--video_path",type=str,default="/home/asad/Downloads/drive_cut_2.mp4")
args=parser.parse_args()

cap = cv2.VideoCapture(args.video_path)

model=unetModel()
unet=model.get_unet()
unet.load_weights("best_unet_lane.h5")


def main():
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            rgb=frame[...,::-1]
            rgb= tf.convert_to_tensor(rgb, dtype=tf.float32)
            t_input=process_img(rgb)
            intial_pred=create_mask(unet.predict(t_input))
            cv_img=np.array(tf.keras.preprocessing.image.array_to_img(intial_pred[0]))
            rgb_pred=cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            rgb_pred[...,1]*=2
            rgb_pred[...,2]*=3
            cv2.imshow('Frame',rgb_pred)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

cap.release()
cv2.destroyAllWindows()


if __name__=="__main__":
    main()