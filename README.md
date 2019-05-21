# pos-tagger

This project is a pos-tagger.
It determines the part of speech of each token of a given text.

It works with supervised machine learning.
The core algorithm is called perceptron. It is the first neural network ever designed.
It needs a learning corpus with parts of speech already assigned.

- - - - - - - - - - - - - - - - 

The data files consist in a list of sentences. Each sentence is a list of words, and comes with a list of the corresponding tags. Here is the first sentence of the corpus as an example :
[['Les', 'commotions', 'cérébrales', 'sont', 'devenu', 'si', 'courantes', 'dans', 'ce', 'sport', "qu'", 'on', 'les', 'considére', 'presque', 'comme', 'la', 'routine', '.'], ['DET', 'NOUN', 'ADJ', 'AUX', 'VERB', 'ADV', 'ADJ', 'ADP', 'DET', 'NOUN', 'SCONJ', 'PRON', 'PRON', 'VERB', 'ADV', 'ADV', 'DET', 'NOUN', 'PUNCT']]

The data is in french.

For classifying each word, the features are :
- first letter is uppercase
- whole word is uppercase

The classification also includes surrounding words in a window of 2 (left and right, total words = 5) and suffixes.
Features can be found in file test_features.py , function get_features .

