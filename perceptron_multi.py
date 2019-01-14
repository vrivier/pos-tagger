import json
from collections import defaultdict


def get_features(sentence, word_idx):
    res = defaultdict(int)
    if sentence[word_idx][0].isupper():
        res["maj " + sentence[word_idx]] += 1
    if sentence[word_idx].isupper():
        res["all_maj " + sentence[word_idx]] += 1
    if word_idx > 2:
        res["moins_deux " + sentence[word_idx - 2]] += 1
    if word_idx > 1:
        res["moins_un " + sentence[word_idx - 1]] += 1
        if len(sentence[word_idx - 1]) > 3:
            res["suf_moins_un " + sentence[word_idx - 1][-3:]] += 1
    res["zero " + sentence[word_idx]] += 1
    if len(sentence[word_idx]) > 3:
        res["suf_zero " + sentence[word_idx][-3:]] += 1
    if word_idx < len(sentence) - 1:
        res["plus_un " + sentence[word_idx + 1]] += 1
        if len(sentence[word_idx + 1]) > 3:
            res["suf_plus_un " + sentence[word_idx + 1][-3:]] += 1
    if word_idx < len(sentence) - 2:
        res["plus_deux " + sentence[word_idx + 2]] += 1
    return res


# scalaire entre dictionnaires
def dot(w, x):
    res = 0
    for f in x:
        if f in w:
            res += x[f] * w[f]
    return res


class Perceptron:

    def __init__(self):
        self.c = dict()
        self.updates = 0

    def class_vectors(self, corpus):
        res = dict()
        classes = set()
        for s in corpus:
            word_idx = 0
            for w in s[0]:
                if w[0].isupper():
                    res["maj " + s[0][word_idx]] = 0
                if w.isupper():
                    res["all_maj " + s[0][word_idx]] = 0
                if word_idx > 1:
                    res["moins_deux " + s[0][word_idx - 2]] = 0
                if word_idx > 0:
                    res["moins_un " + s[0][word_idx - 1]] = 0
                    if len(s[0][word_idx - 1]) > 3:
                        res["suf_moins_un " + s[0][word_idx - 1][-3:]] = 0
                res["zero " + s[0][word_idx]] = 0
                if len(s[0][word_idx]) > 3:
                    res["suf_zero " + s[0][word_idx][-3:]] = 0
                if word_idx < len(s[0]) - 1:
                    res["plus_un " + s[0][word_idx + 1]] = 0
                    if len(s[0][word_idx + 1]) > 3:
                        res["suf_plus_un " + s[0][word_idx + 1][-3:]] = 0
                if word_idx < len(s[0]) - 2:
                    res["plus_deux " + s[0][word_idx + 2]] = 0
                classes.add(s[1][word_idx])
                word_idx += 1
        for c in classes:
            self.c[c] = res

    def classify(self, x):
        res = list(self.c)[0]
        max = dot(self.c[res], x)
        for y in self.c:  # y c'est la classe
            wy = self.c[y]  # wy c'est son vecteur associé
            if dot(wy, x) > max:
                max = dot(wy, x)
                res = y
        return res

    def update(self, x, label, y):
        for f in x:
            self.c[label][f] += x[f]
            self.c[y][f] -= x[f]

    def train(self, corpus):
        for sentence in corpus:
            for word_idx in range(len(sentence[0])):
                x = get_features(sentence[0], word_idx)
                label = sentence[1][word_idx]
                y = self.classify(x)
                if y != label:
                    self.update(x, label, y)
                    self.updates += 1

    def evaluate(self, corpus):
        res = 0
        nb_words = 0
        for sentence in corpus:
            for word_idx in range(len(sentence[0])):
                nb_words += 1
                x = get_features(sentence[0], word_idx)
                label = sentence[1][word_idx]
                y = self.classify(x)
                if y == label:
                    res += 1
        return res / nb_words


train, dev, test = [json.loads(open("pos.fr.ud13.{}.json".format(corpus)).read()) for corpus in
                    ["train", "dev", "test"]]
# print(get_features(train[0][0],0))
# train=[[['Le','petit','chat','dort'],['DET','ADJ','NOUN','VERB']]]
p = Perceptron()
p.class_vectors(train)
p.train(train)
print(p.evaluate(train))

odd = [list(zip(*[(fields[0], fields[1]) for fields in s])) for s in json.loads(open("fr.foot.json").read())]
