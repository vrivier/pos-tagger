from collections import defaultdict


# scalaire entre dictionnaires
def dot(w, x):
    res = 0
    for f in x:
        if f in w:
            res += w[f]
    return res


class Perceptron:

    def __init__(self, X, y):
        self.X = X # liste de dictionnaires {feature : value}
        self.y = y
        self.C = dict()
        self.init_weights() # dictionnaire {classe : poids}

    def init_weights(self):
        for classe in set(self.y):
            self.C[classe] = defaultdict(int)

    def classify(self, x):
        """ renvoie le plus haut scalaire entre le vecteur de l'exemple et les vecteurs de poids des classes """
        res = 0
        max = -1e6
        for y in self.C:  # y = une classe
            wy = self.C[y]  # wy = son vecteur de poids
            if dot(wy, x) > max:
                max = dot(wy, x)
                res = y
        return res

    def update(self, x, label, pred):
        for f in x:
            self.C[label][f] += x[f]
            self.C[pred][f] -= x[f]

    def train(self, epoque=1):
        for j in range(epoque):
            for i in range(len(self.X)):
                x = self.X[i]
                pred = self.classify(x)
                gold = self.y[i]
                if pred != self.y[i]:
                    self.update(x, gold, pred)

    def evaluate(self, X, y):
        res=0
        for i in range(len(X)):
            pred = self.classify(X[i])
            if pred == y[i]:
                res+=1
        return res/len(X)




