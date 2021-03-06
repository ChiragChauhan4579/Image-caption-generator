from pickle import load,dump
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical,plot_model
import numpy as np
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

"""# Model"""

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_set(filename):
    doc = load_doc(filename)
    dataset = []
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = []
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features

filename = '/content/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: ',len(train))
train_descriptions = load_clean_descriptions('/content/descriptions.txt', train)
print('Descriptions: train=',len(train_descriptions))
train_features = load_photo_features('/content/features.pkl', train)
print('Photos: train=',len(train_features))

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: ',vocab_size)

def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

"""
Here is a example of how our train and test sets would look like after sequences are created
X1,     X2 (text sequence),                         y (word)
photo   startseq,                                   little
photo   startseq, little,                           girl
photo   startseq, little, girl,                     running
photo   startseq, little, girl, running,            in
photo   startseq, little, girl, running, in,        field
photo   startseq, little, girl, running, in, field, endseq
"""

def define_model(vocab_size, max_length):
    inputs1 = tf.keras.layers.Input(shape=(1000,))
    fe1 = tf.keras.layers.Dropout(0.5)(inputs1)
    fe2 = tf.keras.layers.Dense(256, activation='relu')(fe1)
    inputs2 = tf.keras.layers.Input(shape=(max_length,))
    se1 = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = tf.keras.layers.Dropout(0.5)(se1)
    se3 = tf.keras.layers.LSTM(256)(se2)
    decoder1 = tf.keras.layers.add([fe2, se3])
    decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)
    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

filename = '/content/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: ',len(train))
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=',len(train_descriptions))
train_features = load_photo_features('features.pkl', train)
print('Photos: train=',len(train_features))
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: ',vocab_size)
max_length = max_length(train_descriptions)
print('Description Length: ',max_length)
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions,train_features)
filename = '/content/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: ',len(test))
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=',len(test_descriptions))
test_features = load_photo_features('features.pkl', test)
print('Photos: test=',len(test_features))
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions,test_features)
model = define_model(vocab_size, max_length)

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=1,save_best_only=True, mode='min')
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint],validation_data=([X1test, X2test], ytest))

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def cleanup_summary(summary):
    index = summary.find('startseq ')
    if index > -1:
        summary = summary[len('startseq '):]
    index = summary.find(' endseq')
    if index > -1:
        summary = summary[:index]
    return summary

"""# Evaluation"""

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()

    for key, desc_list in descriptions.items():
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        yhat = cleanup_summary(yhat)
        references = [cleanup_summary(d).split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())

    print('BLEU-1: ',corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: ',corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: ',corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: ',corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

filename = '/content/model.h5'
model_trial = load_model(filename)
evaluate_model(model_trial, test_descriptions, test_features, tokenizer, max_length)

dump(tokenizer, open('tokenizer.pkl', 'wb'))

import cv2
from google.colab.patches import cv2_imshow

def extract_features(filename):
    model = VGG16()
    model.layers.pop()
    model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
      if index == integer:
        return word
    return None

def cleanup_summary(summary):
    index = summary.find('startseq ')
    if index > -1:
        summary = summary[len('startseq '):]
    index = summary.find(' endseq')
    if index > -1:
        summary = summary[:index]
    return summary

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

tokenizer = load(open('/content/tokenizer.pkl', 'rb'))
max_length = 30
model = load_model('/content/model.h5')
photo = extract_features('/content/photo.jfif')
description = generate_desc(model, tokenizer, photo, max_length)
description = cleanup_summary(description)
img = cv2.imread('/content/photo.jfif')
img = cv2.resize(img,(300,300))
cv2_imshow(img)
print(description)

"""Need to train the model a lot more to make predictions better"""
