import sys
import os
import json
import pandas
import numpy
import optparse
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import csv
from keras import backend as K

def predict(csv_file):
    # Loading processed word dictionary into keras Tokenizer would be better
    dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
    dataset = dataframe.values

    # Preprocess dataset
    X = dataset[:,0]
    Y = dataset[:,1]
    
    for index, item in enumerate(X):
        reqJson = json.loads(item, object_pairs_hook=OrderedDict, strict=False)
        X[index] = json.dumps(reqJson, separators=(',', ':'))
        
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X)
    seq = tokenizer.texts_to_sequences(X)
    max_log_length = 1024
    log_entry_processed = sequence.pad_sequences(seq, maxlen=max_log_length)


    model = load_model('malicious-requests-model.h5')
    model.load_weights('malicious-requests-weights.h5')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    emb = model.predict(log_entry_processed)

    embedding_var = tf.Variable(emb)
    LOG_DIR = './logs'
    
    print(emb)
    
    length = int(round(len(X)* .05))
    end = int(round(len(X) - len(X) * .05))
    print(length)
    print(end)
    print(len(X) - 1)
    aList = []
    for i in range(length):
        aList.append([i,emb[i][0]])

    for i in range(end, len(X) - 1):
        aList.append([i,emb[i][0]])
    images = tf.Variable(aList, name='requests')
    with tf.Session() as sess:
        saver = tf.train.Saver([images])

        sess.run(images.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'requests.ckpt'))
    
    word_dict_file = './logs/metadata.tsv'
    with open(word_dict_file, 'w') as outfile:
        w = csv.writer(outfile, quoting = csv.QUOTE_NONE, delimiter='|', quotechar='',escapechar='\\')
        w.writerow(["{0}\t{1}\t{2}".format('Index', 'Label', 'Class')])
        for i in range(length):
            w.writerow(['{0}\t{1}\t{2}'.format(i, X[i], Y[i])])
        for i in range(end, len(X) - 1):
            w.writerow(['{0}\t{1}\t{2}'.format(i, X[i], Y[i])])


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/training.csv'

    predict(csv_file)
