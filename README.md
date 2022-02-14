# Image-caption-generator

A project that generates caption for the given input image.

Dataset used:- Flickr8K

Framework used:- Tensorflow

## Documentation

The project workflow is divided into two portions:-
1. Generating features from images and cleaning descriptions
2. Modelling and evaluation using bleu scores


The features are extracted using VGG model and stored in a pickle format. While the descriptions are saved as txt file.

The model used is a simple LSTM model.

For more info on bleu scores:-
1. https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
2. https://towardsdatascience.com/bleu-bilingual-evaluation-understudy-2b4eab9bcfd1
