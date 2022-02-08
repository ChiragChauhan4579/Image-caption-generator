#Download the dataset by running these commands in cmd
"""
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
"""

from zipfile import ZipFile
file_name = "Flickr8k_Dataset.zip" #Do give a proper directory
with ZipFile(file_name, 'r') as zip:
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')

from zipfile import ZipFile
file_name = "Flickr8k_text.zip" #Do give a proper directory
with ZipFile(file_name, 'r') as zip:
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')

from os import listdir,path
from pickle import dump
import re
import string
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

def extract_features(directory):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    model.summary()
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
    return features

directory = '/content/Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
dump(features, open('features.pkl', 'wb'))

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

filename = '/content/Flickr8k.token.txt'
doc = load_doc(filename)

def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
            mapping[image_id].append(image_desc)
    return mapping

descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))

def clean_descriptions(descriptions):
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [re_punc.sub('', w) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)
            
clean_descriptions(descriptions)

def to_vocabulary(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))

def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
            data = '\n'.join(lines)
            file = open(filename, 'w')
            file.write(data)
            file.close()

save_descriptions(descriptions, 'descriptions.txt')