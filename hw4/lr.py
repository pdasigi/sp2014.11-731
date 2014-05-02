
import math, sys, random

class LogisticRegression:
    def __init__(self):
        self.rate = 0.01
        self.weight = {}
        return
    def train(self, data, n):
        for i in range(n):
            updates = []
            for [label, feature] in data:
                predicted = self.classify(feature)
                for f,v in feature.iteritems():
                    if f not in self.weight:
                        self.weight[f] = 0
                    update = (label - predicted) * v
                    updates.append(update)
                    self.weight[f] = self.weight[f] + self.rate * update
            print >>sys.stderr, 'iteration', i, 'done', 'new weights', self.weight
        return
    def classify(self, feature):
        logit = 0
        for f,v in feature.iteritems():
            coef = 0
            if f not in self.weight:
                self.weight[f] = random.random()
            coef = self.weight[f]
            logit += coef * v
        try:
            return 1.0 / (1.0 + math.exp(-logit))
        except:
            print >>sys.stderr, "Incompatible logit", logit
            return 1
