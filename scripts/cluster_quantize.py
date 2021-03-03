import tensorflow as tf
from utils import dataset

lane_data=dataset("/media/asad/8800F79D00F79104/lanes_data/20k_images/","/media/asad/8800F79D00F79104/lanes_data/20k_labels/")
datasets=lane_data.load_dataset(b_sz=1)

def representative_dataset():
  for data,label_list in datasets["val"].take(100):
    yield [data]


#rep_data=representative_dataset()
 #for item in rep_data:
 #   print(item)

striped_clustered_model=tf.keras.models.load_model("striped_clustered.h5")
striped_clustered_model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(striped_clustered_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset
tflite_quant_model = converter.convert()

#_, quantized_and_clustered_tflite_file = tempfile.mkstemp('.tflitecluster')

with open("cluster_quantized_weights.tflite", 'wb') as f:
    f.write(tflite_quant_model)