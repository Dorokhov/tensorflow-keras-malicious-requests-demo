import sys
import os
import json
import pandas
import numpy
import optparse
import tensorflow as tf
import csv

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
from keras.backend import shape
from keras import backend as K

from tensorflow.contrib.tensorboard.plugins import projector
from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy

class TensorResponseBoard(TensorBoard):
    def __init__(self, val_size, **kwargs):
        super(TensorResponseBoard, self).__init__(**kwargs)
        self.val_size = val_size
        #self.img_path = img_path
        #self.img_size = img_size

    def set_model(self, model):
        super(TensorResponseBoard, self).set_model(model)

        if self.embeddings_freq and self.embeddings_layer_names:
            embeddings = {}
            for layer_name in self.embeddings_layer_names:
                # initialize tensors which will later be used in `on_epoch_end()` to
                # store the response values by feeding the val data through the model
                layer = self.model.get_layer(layer_name)
                output_dim = layer.output.shape[-1]
                response_tensor = tf.Variable(tf.zeros([self.val_size, output_dim]),
                                              name=layer_name + '_response')
                embeddings[layer_name] = response_tensor

            self.embeddings = embeddings
            self.saver = tf.train.Saver(list(self.embeddings.values()))

            response_outputs = [self.model.get_layer(layer_name).output
                                for layer_name in self.embeddings_layer_names]
            self.response_model = Model(self.model.inputs, response_outputs)

            config = projector.ProjectorConfig()
            embeddings_metadata = {layer_name: self.embeddings_metadata
                                   for layer_name in embeddings.keys()}

            for layer_name, response_tensor in self.embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = response_tensor.name

                # for coloring points by labels
                embedding.metadata_path = embeddings_metadata[layer_name]

                # for attaching images to the points
                #embedding.sprite.image_path = self.img_path
                # embedding.sprite.single_image_dim.extend(self.img_size)

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        super(TensorResponseBoard, self).on_epoch_end(epoch, logs)

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                # feeding the validation data through the model
                val_data = self.validation_data[0]
                response_values = self.response_model.predict(val_data)

                numpy.set_printoptions(threshold=sys.maxsize)
                print ("PREDICT")
                print (val_data[0])
                print ( response_values[0])
                
                if len(self.embeddings_layer_names) == 1:
                    response_values = [response_values]

                # record the response at each layers we're monitoring
                response_tensors = []
                for layer_name in self.embeddings_layer_names:
                    print (layer_name)
                    response_tensors.append(self.embeddings[layer_name])
                K.batch_set_value(list(zip(response_tensors, response_values)))
                
                # finally, save all tensors holding the layer responses
                print (self.embeddings_ckpt_path)
                self.saver.save(self.sess, self.embeddings_ckpt_path, epoch)

def train(csv_file):
    dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
    dataset = dataframe.sample(frac=1).values

    # Preprocess dataset
    X = dataset[:,0]
    Y = dataset[:,1]

    for index, item in enumerate(X):
        reqJson = json.loads(item, object_pairs_hook=OrderedDict, strict=False)
        X[index] = json.dumps(reqJson, separators=(',', ':'))

    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    print(X)
    tokenizer.fit_on_texts(X)

    # Extract and save word dictionary
    word_dict_file = 'logs/metadata.tsv'

    if not os.path.exists(os.path.dirname(word_dict_file)):
        os.makedirs(os.path.dirname(word_dict_file))
    print("A")
    print(tokenizer.word_index)
    print(type(tokenizer.word_index))
    with open(word_dict_file, 'w') as outfile:
        #json.dump(tokenizer.word_index, outfile, ensure_ascii=False)
        w = csv.writer(outfile)
        w.writerow(["{0}\t{1}\t{2}".format('Text', 'Index', 'Class')])
        w.writerow(["{0}\t{1}\t{2}".format('0', '0', '0')])
        for key, val in tokenizer.word_index.items():
            w.writerow(["{0}\t{1}\t{2}".format(key, val, ('0' if val % 2  else '1'))])

    num_words = len(tokenizer.word_index)+1
    
    X = tokenizer.texts_to_sequences(X)
    
    max_log_length = 1024
    train_size = int(len(dataset) * .9)

    X_processed = sequence.pad_sequences(X, maxlen=max_log_length)
    X_train, X_test = X_processed[0:train_size], X_processed[train_size:len(X_processed)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    LOG_DIR = './logs'
    print("WORDS")
    print(X_train.shape)
    print(Y_train.shape)
                      
    model = Sequential() 
    model.add(Embedding(num_words, 64, input_length=max_log_length))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid', use_bias=False))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    embedding_layer_names = set(layer.name
                            for layer in model.layers
                            if layer.name.startswith('dense_'))
    #tb_callback = TensorResponseBoard(log_dir=LOG_DIR,val_size=893, embeddings_metadata='metadata.tsv', embeddings_freq=1, embeddings_layer_names=embedding_layer_names) #
    tb_callback = TensorBoard(log_dir=LOG_DIR, embeddings_freq=1) #, embeddings_layer_names=embedding_layer_names
    print(embedding_layer_names)
    model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=3, callbacks=[tb_callback])

    # Evaluate model
    score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=128)

    print("accuracy: {:0.2f}%".format(acc * 100))

    # Config
    LOG_DIR = 'logs'
    embedding_var = tf.Variable(5, name='requests')
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = 'metadata.tsv'

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)

    # Save model
    model.save_weights('malicious-requests-weights.h5')
    model.save('malicious-requests-model.h5')
    with open('malicious-requests-model.json', 'w') as outfile:
        outfile.write(model.to_json())



if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/training.csv'
    train(csv_file)
