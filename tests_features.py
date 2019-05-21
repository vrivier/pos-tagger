import json
import perceptron_dic as pdic


def get_features(sentence, word_idx):
    """ renvoie la liste des features d'un mot """
    res = dict()
    if sentence[word_idx][0].isupper():
        res["maj " + sentence[word_idx]] = 1
    if sentence[word_idx].isupper():
        res["all_maj " + sentence[word_idx]] = 1
    if word_idx > 2:
        res["moins_deux " + sentence[word_idx - 2]] = 1
    if word_idx > 1:
        res["moins_un " + sentence[word_idx - 1]] = 1
        if len(sentence[word_idx - 1]) > 3:
            res["suf_moins_un " + sentence[word_idx - 1][-3:]] = 1
    res["zero " + sentence[word_idx]] = 1
    if len(sentence[word_idx]) > 3:
        res["suf_zero " + sentence[word_idx][-3:]] = 1
    if word_idx < len(sentence) - 1:
        res["plus_un " + sentence[word_idx + 1]] = 1
        if len(sentence[word_idx + 1]) > 3:
            res["suf_plus_un " + sentence[word_idx + 1][-3:]] = 1
    if word_idx < len(sentence) - 2:
        res["plus_deux " + sentence[word_idx + 2]] = 1
    return res


def prepare(corpus):
	exemples = list()
	classes_gold = list()

	for sentence in train:
	    for word_idx in range(len(sentence[0])):
	        x = get_features(sentence[0], word_idx)
	        exemples.append(x)
	        classes_gold.append(sentence[1][word_idx])

	return (exemples, classes_gold)


train, dev, test = [json.loads(open("pos.fr.ud13.{}.json".format(corpus)).read()) for corpus in
                    ["train", "dev", "test"]]

train, dev, test = [prepare(x) for x in [train, dev, test]]


p = pdic.Perceptron(train[0], train[1])
p.train()
score = p.evaluate(test[0], test[1])
print(score)

