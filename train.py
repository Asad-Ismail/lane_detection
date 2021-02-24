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
import numpy as np
from losses import custom_sparse_weighted_crossentropy

parser=argparse.ArgumentParser(description="Input Args")
#parser.add_argument("--train_images",type=str,default="/media/asad/8800F79D00F79104/lanes_data/images/")
#parser.add_argument("--train_labels",type=str,default="/media/asad/8800F79D00F79104/lanes_data/labels/")

parser.add_argument("--train_images",type=str,default="/media/asad/8800F79D00F79104/lanes_data/20k_images/")
parser.add_argument("--train_labels",type=str,default="/media/asad/8800F79D00F79104/lanes_data/20k_labels/")

args=parser.parse_args()


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


BATCH_SIZE=3
EPOCHS = 100
ClASSES=5 # BAckgroud plus four lanes
lr=0.0001

def main():
    mydata=dataset(args.train_images,args.train_labels)
    datasets=mydata.load_dataset()
    # Test and plot data
    #data=list(datasets["train"].take(1).as_numpy_iterator())
    #sample_image,sample_label=data[0][0],data[0][1]
    #display_sample([sample_image,sample_label])
    # Model declration 
    model=unetModel(fine_tune=True)
    unet=model.get_unet()
    #print(unet.summary())
    #print(f"Loading pretrained model")
    unet.load_weights("best_unet_lane.h5")
    unet.trainable=True
    #tf.keras.utils.plot_model(unet, show_shapes=True)
    mIOU = tf.keras.metrics.MeanIoU(num_classes=ClASSES)
    class_weights=[1.0,1,1,1,1] 
    loss=custom_sparse_weighted_crossentropy(class_weights)  
    #loss=tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)
    unet.compile(optimizer=Adam(learning_rate=lr), loss =loss,metrics=['accuracy'])    
    #sample show predictions
    #intial_pred=create_mask(unet.predict(sample_image))
    #show_predictions(sample_image=sample_image,sample_label=sample_label,sample_pred=intial_pred)
    EPOCHS = 100
    BATCH_SIZE=3
    STEPS_PER_EPOCH = mydata.train_size // BATCH_SIZE
    VALIDATION_STEPS = mydata.val_size // BATCH_SIZE
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    callbacks = [
    # to show samples after each epoch
    #DisplayCallback(),
    # to collect some useful metrics and visualize them in tensorboard
    tensorboard_callback,
    # if no accuracy improvements we can stop the training directly
    tf.keras.callbacks.EarlyStopping(patience=5, verbose=1),
    # to save checkpoints
    tf.keras.callbacks.ModelCheckpoint('best_unet_lane_big.h5', verbose=1, save_best_only=True,monitor='val_accuracy', save_weights_only=True)
    ]
    
    model_history = unet.fit(datasets['train'], epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=datasets['val'],
                    callbacks=callbacks)


if __name__=="__main__":
    main()