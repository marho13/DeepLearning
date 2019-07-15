import re, sys, os, math, random
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse

random.seed(42)

epochs = 1000

def readAndWrite(file):
    data = open(file, "r").read()
    data = data.split("\n")
    split = [d.split(",") for d in data]
    del split[0]
    del split[-1][-1]
    writer = open("this.csv", "w")
    sumy = [0]*15
    for a in range(len(split)):
        for b in range(len(split[a])):
            if float(split[a][b]) > sumy[b]: sumy[b]=round(float(split[a][b]))

    for c in range(len(split)):
        for d in range(len(split[c])):
            if d!=1:
                split[c][d] = float(split[c][d])/sumy[d]

    for a in range(len(split)):
        if a != 0:
            for b in range(len(split[a])):
                if b == 1:
                    writer.write(str(round(float(split[a][b]))))
                    writer.write(",")
                else:
                    writer.write(str(split[a][b]))
                    writer.write(",")
            writer.write("\n")
    writer.close()

    # file = open("this.csv", "w")
    # file.write([d for d in data.split("\n")])

class fileRead:
    def __init__(self, file, testRatio):
        #Variable Initiation
        self.ratio = testRatio
        self.fileName = file

        #Read data into dataset list, then find length of data
        data = open(self.fileName, "r").read()
        data = data.split("\n")

        self.dataset, self.headers = self.read(data)
        self.length = len(self.dataset)

        #Shuffle said list
        self.shuffleData()

        self.train, self.test = self.splitTrainandTest()
        # Split the data into testing/training labels and logits
        self.trainLogit, self.trainLabel = self.labelLogit(self.train)
        self.testLogit, self.testLabel = self.labelLogit(self.test)

    def numCleanup(self, x):
        return float("{:.2f}".format(x))

    def read(self, data):
        header = data.pop(0).split(",")
        #Creates the header, and dataset
        dataset = [d.split(",") for d in data]

        #Removes inevitable empty entry at the last entry
        del dataset[-1]
        del dataset[-1]

        for a in range(len(dataset)):
            del dataset[a][-1]
        return dataset, header

    def shuffleData(self):
        random.shuffle(self.dataset)

    def shuffleTrain(self):
        random.shuffle(self.train)
        self.trainLogit, self.trainLabel = self.labelLogit(self.train)

    def splitTrainandTest(self):
        return self.dataset[int(self.length*self.ratio):], self.dataset[:int(self.length*self.ratio)]

    def labelLogit(self, dataset):
        label, logit = [], []
        for a in range(len(dataset)):
            logit.append([])
            for b in range(len(dataset[a])):
                if b == 1:
                    label.append([self.numCleanup(float(dataset[a][b]))])
                else:
                    logit[-1].append([self.numCleanup(float(dataset[a][b]))])
        return logit, label

class backprop:
    def __init__(self, lr, lambda1, lambda2):
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        pass

    def apply_gradients(self, weights, change):
        for x in range(len(change)):
            for y in range(len(change[x])):
                change[x][y] *= self.lr
        weights = weights - change
        return weights

    def minimize(self):
        pass

class BCD:
    def __init__(self, lr, lambda1, lambda2):
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def apply_gradients(self, weights, change):
        for x in range(len(change)):
            for y in range(len(change[x])):
                change[x][y] *= self.lr
        weights = weights - change
        return weights

    def minimize(self):
        pass

class ADAM:
    def __init__(self, lr, beta1, beta2, lambda1, lambda2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def calcMoment(self, mt, beta1, grad):
        mt = beta1 * mt + (1-beta1)*grad

    def calcVelocity(self, vt, beta2, grad):
        vt = beta2 * vt + (1-beta2)*(grad*grad)

    def apply_gradients(self, weights, change):
        for x in range(len(change)):
            for y in range(len(change[x])):
                change[x][y] *= self.lr
        weights = weights - change
        return weights

    def minimize(self):
        pass

class SGD:
    def __init__(self, lr, lambda1, lambda2):
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def apply_gradients(self, weights, change):
        for x in range(len(change)):
            for y in range(len(change[x])):
                change[x][y] *= self.lr
        weights = weights - change
        return weights

    def minimize(self, loss): #Add operation to minimize loss by updating var_list
        pass #if we do the minimization here, that means we need to get the weights of the network

def compute_gradient(x, y, w1, w2, n, lambda1, lambda2):
    W1Change = None
    W2Change = None
    x = np.asarray(x)
    y = np.asarray(y)
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    w1t = np.transpose(w1)
    w2t = np.transpose(w2)
    for a in range(len(x)):
        xt = np.transpose(x[a])
        step11 = (2/n)*(np.matmul(w2t, np.matmul(w2, np.matmul(w1, np.matmul(x[a], xt)))))
        step21 = (2/n)*(np.matmul(w2t, (xt*y[a])))
        step31 = 2*lambda1*w1

        step12 = (2/n)*(np.matmul(w2, np.matmul(w1, np.matmul(x[a], np.matmul(xt, w1t)))))
        step22 = (2/n)*(np.matmul(xt, w1t)*y[a])
        step32 = 2*lambda2*w2

        W1Change = step11 - step21 + step31
        W2Change = step12 - step22 + step32
    return W1Change/len(x), W2Change/len(x)

class LinearNetwork:
    def __init__(self, config, lambda1, lambda2):
        self.networkconfig = config
        self.weights = self.fullyConnectedWeights(config[0], config[1], config[2])
        self.lambda1, self.lambda2 = lambda1, lambda2

    def fullyConnectedWeights(self, inShape, l1Shape, l2Shape):
        weights = [[], []]
        for a in range(l1Shape):
            weights[0].append([])
            for b in range(inShape):
                weights[0][-1].append(0.01)

        for c in range(l2Shape):
            weights[1].append([])
            for d in range(l1Shape):
                weights[1][-1].append(0.01)
        weights = np.asarray(weights)
        return weights

    def backprop(self, method, inputy, ans):
        gradient = compute_gradient(inputy, ans, self.weights[0], self.weights[1], len(inputy), self.lambda1, self.lambda2)
        self.weights = method.apply_gradients(self.weights, gradient)

    def fit(self, inputy, ans, method):
        pred = self.forwardpass(inputy)
        self.backprop(method, inputy, ans)
        return self.mse(pred, ans)


    def predict(self, inputy, ans):
        prediction = self.forwardpass(inputy)
        return self.mse(prediction, ans)

    def formula(self, inputy):
        X1, X2 = None, None
        output = []
        for inp in inputy:
            X1 = self.multiplication(inp, self.weights[0])
            X2 = self.multiplication(X1, self.weights[1])
            output.append(X2)

        return output

    def multiplication(self, inputy, weight):
        inputy = np.asarray(inputy)
        outy = [0]*len(weight)
        for x in range(len(weight)): #len of input
            for y in range(len(weight[x])): #len of weights
                if len(weight)> 1:
                    try:
                        outy[x] += float(weight[x][y])*float(inputy[y][0])
                    except:
                        print(np.asarray(weight).shape, np.asarray(inputy).shape)

                else:
                    outy[x] += float(weight[x][y])*float(inputy[x])
        return outy

    def forwardpass(self, inputy):
        return self.formula(inputy)

    def l2norm(self, inputy):
        sum = 0
        sum += (inputy)** 2
        return math.sqrt(sum)

    def frobNorm(self, inputy):
        sum = 0
        for inp in inputy:
            for i in inp:
                sum += i ** 2
        return math.sqrt(sum)

    def mse(self, inputy, ans):
        sum = 0
        inputy = np.asarray(inputy)
        ans = np.asarray(ans)
        for i in range(len(inputy)):
            sum += mse(inputy[i], ans[i])
        return sum/len(inputy)


def runner(f, epochs=1, batchSize=202, network=LinearNetwork([14, 10, 1], 0.1, 0.2), method=backprop(0.1, 0.1, 0.1)):
    if batchSize < 10: batchSize = 202
    for e in range(epochs):
        loss = 0.0
        f.shuffleTrain()
        trainLogit, trainLabel = f.trainLogit, f.trainLabel
        testingLogit, testingLabel = f.testLogit, f.testLabel
        for a in range((202//batchSize)):
            trainX = np.asarray(trainLogit[a*batchSize: (a+1)*batchSize])
            trainY = np.asarray(trainLabel[a*batchSize: (a+1)*batchSize])
            batchloss = network.fit(trainX, trainY, method)
            loss += batchloss
            #Send the data in, and get the loss out
            #do the backprob, and lastly return loss
        loss /= int(202//batchSize)
        testloss = network.predict(testingLogit, testingLabel)
    print("Epochs done, average loss is {} for training, and {} for testing".format(loss, testloss))
    return loss, testloss

# readAndWrite("hw3 - dataset/Bodyfat.csv")
f = fileRead("this.csv", 0.2)

def iterator(f):
    setup = []
    adamRun = []
    score = []
    setupHeader = ["LearningRate", "BatchSize", "W1 Size"]
    lr = [0.01, 0.001]
    batchSize = [10, 50, 200]
    w1 = [50, 100, 500, 1000]
    lambda1 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    lambda2 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    for l1 in lambda1:
        for l2 in lambda2:
            for size in w1:
                for l in lr:
                    methods = [SGD(l, l1, l2)]
                    for b in batchSize:
                        print(b, l, l1, l2, size)
                        network = LinearNetwork([14, size, 1], l1, l2)
                        for method in methods:
                            setup.append([l, b, size, l1, l2, method])
                            trainLoss, testLoss = runner(f, epochs, b, network, method)
                            score.append([trainLoss, testLoss])
    return setupHeader, setup, score, adamRun


header, setup, score, adam = iterator(f)
for a in range(len(score)):
    print("{} : {}   -   {} : {}   -   {} : {}".format(header[0], setup[a][0], header[1], setup[a][1], header[2], setup[a][2]))
    print(score[a], adam[a])
