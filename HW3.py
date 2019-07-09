import re, sys, os, math, random
import tensorflow as tf

random.seed(42)

learning_rate = 0.01
batchSize = 202
epochs = 100
w1 = 1000

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
                    logit[-1].append(self.numCleanup(float(dataset[a][b])))
        return logit, label


def createModel(X, w1, w2):
    W1 = tf.layers.dense(X, units=w1, activation=tf.nn.sigmoid)
    return tf.layers.dense(W1, units=w2, activation=tf.nn.sigmoid)

def runner(f, opt, inputShapy, outShapy, w1, epochs=1, batchSize=202, lr=0.1):
    max = 0
    if batchSize < 10: batchSize = 202
    X = tf.placeholder(dtype=tf.float32, shape=[None, inputShapy], name="Input")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Output")
    learningRate = tf.placeholder(dtype=tf.float32)
    model = createModel(X, w1, outShapy)
    loss = tf.losses.absolute_difference(model, Y)
    if opt == "Adam":
        optimizer = tf.train.AdamOptimizer()
    else:
        optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for e in range(epochs):
        corr, tot = 0,0
        f.shuffleTrain()
        trainLogit, trainLabel = f.trainLogit, f.trainLabel
        for a in range((202//batchSize)):
            trainX = trainLogit[a*batchSize: (a+1)*batchSize]
            trainY = trainLabel[a*batchSize: (a+1)*batchSize]
            batchLoss, _, pred = sess.run([loss, train, model], feed_dict={X:trainX, Y:trainY, learningRate:lr})
            for a in range(len(pred)):
                if abs(round(pred[a][0]) - round(trainY[a][0])) == 2:
                    corr += 1
                    tot += 1
                else:
                    tot +=1
        if max <= (corr/tot): max = (corr/tot)
    return max


# readAndWrite("hw3 - dataset/Bodyfat.csv")
f = fileRead("this.csv", 0.2)

def iterator(f):
    setup = []
    adamRun = []
    score = []
    setupHeader = ["LearningRate", "BatchSize", "W1 Size"]
    lr = [0.1, 0.01, 0.001, 0.0001]
    batchSize = [10, 50, 202]
    w1 = [10000, 5000, 1000, 500, 100, 50]
    for l in lr:
        for b in batchSize:
            for size in w1:
                setup.append([l, b, size])
                score.append(runner(f, "notAdam", len(f.trainLogit[0]), 1, size, epochs, b, l))
                adamRun.append(runner(f, "Adam", len(f.trainLogit[0]), 1, size, epochs, b, 1))
    return setupHeader, setup, score, adamRun


header, setup, score, adam = iterator(f)
for a in range(len(score)):
    print("{} : {}   -   {} : {}   -   {} : {}".format(header[0], setup[a][0], header[1], setup[a][1], header[2], setup[a][2]))
    print(score[a], adam[a])


# f = fileRead("hw3 - dataset/Bodyfat.csv", 0.2)
# f.shuffleTrain()
# runner(f, "NotAdam", len(f.trainLogit[0]), 1, w1, epochs, batchSize, learning_rate)


# def loss(x, y):
#     pass
#     #perform loss calculation using formula(1)
#
# class backprop:
#     def __init__(self, lr, lambda1, lambda2):
#         self.lr = lr
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         pass
#
#     def compute_gradient(self, x, y, w1, w2):
#         W1Change = None
#         W2Change = None
#         for i in range(len(y)):
#             W1Change += 2*(w2*x[i] - y[i]) + 2*self.lambda1
#             W2Change += 2*(w1*x[i] - y[i]) + 2*self.lambda2
#         W1Change = W1Change / len(y)
#         W2Change = W2Change / len(y)
#         return W1Change, W2Change
#
#     def apply_gradients(self):
#         # w1 = w1 - W1Change*self.lr
#         # w2 = w2 - W1Change*self.lr
#         # return w1, w2
#         pass
#
#
#
#     def minimize(self):
#         pass
#
# class BCD:
#     def __init__(self, lr, lambda1, lambda2):
#         self.lr = lr
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#
#     def compute_gradient(self, x, y, w1, w2):
#         W1Change = None
#         W2Change = None
#         for i in range(len(y)):
#             W1Change += 2 * (w2 * x[i] - y[i]) + 2 * self.lambda1
#             W2Change += 2 * (w1 * x[i] - y[i]) + 2 * self.lambda2
#         W1Change = W1Change/len(y)
#         W2Change = W2Change / len(y)
#         return W1Change, W2Change
#
#     def apply_gradients(self):
#         pass
#
#     def minimize(self):
#         pass
#
# class ADAM:
#     def __init__(self, lr, beta1, beta2, lambda1, lambda2):
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#
#     def calcMoment(self, mt, beta1, grad):
#         mt = beta1 * mt + (1-beta1)*grad
#
#     def calcVelocity(self, vt, beta2, grad):
#         vt = beta2 * vt + (1-beta2)*(grad*grad)
#
#     def compute_gradient(self, x, y, w1, w2):
#         W1Change = None
#         W2Change = None
#         for i in range(len(y)):
#             W1Change += 2 * (w2 * x[i] - y[i]) + 2 * self.lambda1
#             W2Change += 2 * (w1 * x[i] - y[i]) + 2 * self.lambda2
#         return W1Change, W2Change
#
#     def apply_gradients(self, output, lrt, mt, vt, epsilon):
#         output = output - lrt * mt/(math.sqrt(vt)+epsilon)
#
#     def minimize(self):
#         pass
#
# class SGD:
#     def __init__(self, lr, lambda1, lambda2):
#         self.lr = lr
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#
#     def compute_gradient(self, x, y, w1, w2):
#         W1Change = 0.0
#         W2Change = 0.0
#         for i in range(len(x)):
#             W1Change += 2 * (w2 * x[i] - y[i]) + 2 * self.lambda1
#             W2Change += 2 * (w1 * x[i] - y[i]) + 2 * self.lambda2
#         W1Change = W1Change/len(x)
#         W2Change = W2Change/len(x)
#         return W1Change, W2Change
#
#     def apply_gradients(self): #another part of minimize, takes gradients and applies them
#         pass
#
#     def minimize(self, loss): #Add operation to minimize loss by updating var_list
#         pass #if we do the minimization here, that means we need to get the weights of the network
#
#
# class LinearNetwork:
#     def __init__(self, config, lambda1, lambda2):
#         self.networkconfig = config
#         self.weights = self.fullyConnectedWeights(config[0], config[1], config[2])
#         self.lambda1, self.lambda2 = lambda1, lambda2
#
#     def fullyConnectedWeights(self, inShape, l1Shape, l2Shape):
#         weights = [[], []]
#         for a in range(l1Shape):
#             weights[0].append([])
#             for b in range(inShape):
#                 weights[0][-1].append(random.random())
#
#         for c in range(l2Shape):
#             weights[1].append([])
#             for d in range(l1Shape):
#                 weights[1][-1].append(random.random())
#
#         return weights
#
#
#     def fit(self, backprop, input):
#         pass
#         #Fit send the data in to the network, and do the activation and such
#         #then use the backpropagation method, and so on
#
#     def predict(self, inputy, ans):
#         prediction = self.forwardpass(inputy, ans)
#         return prediction
#
#     def formula(self, inputy, weights, ans, lambda1, lambda2):
#         part1 = 0
#         for inp in inputy:
#             X1 = self.multiplication(inputy, weights[0])
#             X2 = self.multiplication(X1, weights[1])
#
#         return X1, X2
#
#     def multiplication(self, inputy, weight):
#         outy = [0]*len(weight)
#         print(len(inputy), len(weight), len(weight[0]))
#         for x in range(len(weight)): #len of input
#             for y in range(len(weight[x])): #len of weights
#                 if len(weight)> 1:
#                     outy[x] += float(weight[x][y])*float(inputy[x])
#                 else:
#                     outy[x] += float(weight[x][y])*float(inputy[y])
#         return outy
#
#     def forwardpass(self, inputy, ans):
#         return self.formula(inputy, self.weights, ans, self.lambda1, self.lambda2)
#
#     def l2norm(self, inputy, ans):
#         sum = 0
#         sum += (inputy - ans) ** 2
#         return math.sqrt(sum)
#
#     def frobNorm(self, inputy):
#         sum = 0
#         for inp in inputy:
#             for i in inp:
#                 sum += i ** 2
#         return math.sqrt(sum)
#
#
# #Create an instance of the fileRead class
# x = fileRead("hw3 - dataset/Bodyfat.csv", 0.2)
#
# network = LinearNetwork([14, 10, 1], 0.1, 0.2)
# weight = network.weights
# # [print(f, end=", ") for f in x.train[0]]
# # print()
# [print(w) for w in weight]
# X1, X2 = network.formula(x.trainLogit[-1], weight, 0.1, 0.2)
# print(X2)
