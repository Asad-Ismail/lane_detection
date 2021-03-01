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



parser=argparse.ArgumentParser(description="Input Args")
parser.add_argument("--video_path",type=str,default="/home/asad/Downloads/drive_cut.mp4")
args=parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
out_video = cv2.VideoWriter('drive_cut_pred_lite_vs_tf.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,384))


class pred_tf():
    def __init__(self,weights="/media/asad/8800F79D00F79104/best_unet_lane_20k_default_ce.h5"):
        self.model=unetModel()
        self.unet=self.model.get_unet()
        self.unet.trainable=True
        self.unet.load_weights(weights)
    def __call__(self,test_image):
        intial_pred=create_mask(self.unet.predict(test_image))
        cv_img=np.array(tf.keras.preprocessing.image.array_to_img(intial_pred[0]))
        return cv_img
        


class pred_lite():
    def __init__(self,interpreter_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(interpreter_path))
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.output_shape = self.interpreter.get_output_details()[0]["shape"]
        self.interpreter.allocate_tensors()
    def __call__(self,test_image):
        self.interpreter.set_tensor(self.input_index, test_image)
        self.interpreter.invoke()
        output = self.interpreter.tensor(self.output_index)
        intial_pred=create_mask(output())
        lanes = np.array(tf.keras.preprocessing.image.array_to_img(intial_pred[0]))
        return lanes
        

def main():
    tf_model=pred_tf()
    #lite_model=pred_lite("/home/asad/projs/lane_detection/tf-lite-models/model_quant.tflite")
    lite_model=pred_lite("/home/asad/projs/lane_detection/cluster.tflite")
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #Preprocess Image
            frame=cv2.resize(frame,(640,384))
            rgb=frame[...,::-1]
            rgb= tf.convert_to_tensor(rgb, dtype=tf.float32)[tf.newaxis,...]
            rgb/=255.0
            #Forward pass
            lite_lanes=lite_model(rgb)
            tf_lanes=tf_model(rgb)
            #TF lite lane
            rgb_lite=cv2.cvtColor(lite_lanes, cv2.COLOR_GRAY2BGR)
            rgb_lite[...,0]*=2
            rgb_lite[...,1]*=3
            rgb_lite[...,2]*=2
            cv2.putText(rgb_lite, 'Lite Model', (150,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            #tf lanes
            rgb_tf=cv2.cvtColor(tf_lanes, cv2.COLOR_GRAY2BGR)
            rgb_tf[...,0]*=2
            rgb_tf[...,1]*=3
            rgb_tf[...,2]*=2
            cv2.putText(rgb_tf, 'TF Model', (150,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

            alpha=0.4
            beta = (1.0 - alpha)
            blend_lite = cv2.addWeighted(frame, alpha, rgb_lite.astype(np.uint8), beta, 0.0)
            blend_tf = cv2.addWeighted(frame, alpha, rgb_tf.astype(np.uint8), beta, 0.0)
            stacked_image=np.hstack([blend_tf,blend_lite])
            cv2.imshow('Frame',stacked_image)
            out_video.write(stacked_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break
    out_video.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()