
import math, sys

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
            print >>sys.stderr, 'iteration', i, 'done', 'average update', sum(updates)/len(updates)
        return
    def classify(self, feature):
        logit = 0
        for f,v in feature.iteritems():
            coef = 0
            if f in self.weight:
                coef = self.weight[f]
            logit += coef * v
        return 1.0 / (1.0 + math.exp(-logit))
