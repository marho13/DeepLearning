import re, sys, os, math, random

random.seed(42)

#f(x) = L2((W2W1Xi)-Yi) + lambda1(W1)frob + lambda2(W2)frob
#derivative = [2W2loss + 2lambda1, 2W1loss + 2lambda2] where the comma separates w1 and w2s formula (gradient)
#   also derivative is also not a pure derivative but instead a partial derivative with respect to w1, w2
#l2 norm = sum( (yi-( f(xi))**2 ) for all i
#frob norm = sqrt(sum i( sum j(aij**2)))

class fileRead:
    def __init__(self, file, testRatio):
        #Variable Initiation
        self.ratio = testRatio
        self.fileName = file

        #Read data into dataset list, then find length of data
        data = open(self.fileName, "r").read()
        data = data.split("\n")
        self.dataset, self.headers = self.read(data)
        self.length = len(self.dataset)        #Reads the file, then makes a list for each new line

        #Shuffle said list
        self.shuffleData()

        self.train, self.test = self.splitTrainandTest()
        # Split the data into testing/training labels and logits
        self.trainLogit, self.trainLabel = self.labelLogit(self.train)
        self.testLogit, self.testLabel = self.labelLogit(self.test)


    def read(self, data):

        header = data.pop(0).split(",")
        #Creates the header, and dataset
        dataset = [d.split(",") for d in data]

        #Removes inevitable empty entry at the last entry
        del dataset[-1]
        return dataset, header

    def shuffleData(self):
        random.shuffle(self.dataset)

    def splitTrainandTest(self):
        return self.dataset[int(self.length*self.ratio):], self.dataset[:int(self.length*self.ratio)]

    def labelLogit(self, dataset):
        label, logit = [], []
        for a in range(len(dataset)):
            label.append(dataset[a].pop(1))
            logit.append(dataset[a])
        return logit, label



def loss(x, y):
    pass
    #perform loss calculation using formula(1)

class backprop:
    def __init__(self, lr, lambda1, lambda2):
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        pass

    def compute_gradient(self, x, y, w1, w2):
        W1Change = None
        W2Change = None
        for i in range(len(y)):
            W1Change += 2*(w2*x[i] - y[i]) + 2*self.lambda1
            W2Change += 2*(w1*x[i] - y[i]) + 2*self.lambda2
        W1Change = W1Change / len(y)
        W2Change = W2Change / len(y)
        return W1Change, W2Change

    def apply_gradients(self):
        # w1 = w1 - W1Change*self.lr
        # w2 = w2 - W1Change*self.lr
        # return w1, w2
        pass



    def minimize(self):
        pass

class BCD:
    def __init__(self, lr, lambda1, lambda2):
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def compute_gradient(self, x, y, w1, w2):
        W1Change = None
        W2Change = None
        for i in range(len(y)):
            W1Change += 2 * (w2 * x[i] - y[i]) + 2 * self.lambda1
            W2Change += 2 * (w1 * x[i] - y[i]) + 2 * self.lambda2
        W1Change = W1Change/len(y)
        W2Change = W2Change / len(y)
        return W1Change, W2Change

    def apply_gradients(self):
        pass

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

    def compute_gradient(self, x, y, w1, w2):
        W1Change = None
        W2Change = None
        for i in range(len(y)):
            W1Change += 2 * (w2 * x[i] - y[i]) + 2 * self.lambda1
            W2Change += 2 * (w1 * x[i] - y[i]) + 2 * self.lambda2
        return W1Change, W2Change

    def apply_gradients(self, output, lrt, mt, vt, epsilon):
        output = output - lrt * mt/(math.sqrt(vt)+epsilon)

    def minimize(self):
        pass

class SGD:
    def __init__(self, lr, lambda1, lambda2):
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def compute_gradient(self, x, y, w1, w2):
        W1Change = 0.0
        W2Change = 0.0
        for i in range(len(x)):
            W1Change += 2 * (w2 * x[i] - y[i]) + 2 * self.lambda1
            W2Change += 2 * (w1 * x[i] - y[i]) + 2 * self.lambda2
        W1Change = W1Change/len(x)
        W2Change = W2Change/len(x)
        return W1Change, W2Change

    def apply_gradients(self): #another part of minimize, takes gradients and applies them
        pass

    def minimize(self, loss): #Add operation to minimize loss by updating var_list
        pass #if we do the minimization here, that means we need to get the weights of the network


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
                weights[0][-1].append(random.random())

        for c in range(l2Shape):
            weights[1].append([])
            for d in range(l1Shape):
                weights[1][-1].append(random.random())

        return weights


    def fit(self, backprop, input):
        pass
        #Fit send the data in to the network, and do the activation and such
        #then use the backpropagation method, and so on

    def predict(self, inputy, ans):
        prediction = self.forwardpass(inputy, ans)
        return prediction

    def formula(self, inputy, weights, ans, lambda1, lambda2):
        part1 = 0
        for inp in inputy:
            X1 = self.multiplication(inputy, weights[0])
            X2 = self.multiplication(X1, weights[1])

        return X1, X2

    def multiplication(self, inputy, weight):
        outy = [0]*len(weight)
        print(len(inputy), len(weight), len(weight[0]))
        for x in range(len(weight)): #len of input
            for y in range(len(weight[x])): #len of weights
                if len(weight)> 1:
                    outy[x] += float(weight[x][y])*float(inputy[x])
                else:
                    outy[x] += float(weight[x][y])*float(inputy[y])
        return outy

    def forwardpass(self, inputy, ans):
        return self.formula(inputy, self.weights, ans, self.lambda1, self.lambda2)

    def l2norm(self, inputy, ans):
        sum = 0
        sum += (inputy - ans) ** 2
        return math.sqrt(sum)

    def frobNorm(self, inputy):
        sum = 0
        for inp in inputy:
            for i in inp:
                sum += i ** 2
        return math.sqrt(sum)


#Create an instance of the fileRead class
x = fileRead("hw3 - dataset/Bodyfat.csv", 0.2)

network = LinearNetwork([14, 10, 1], 0.1, 0.2)
weight = network.weights
# [print(f, end=", ") for f in x.train[0]]
# print()
[print(w) for w in weight]
X1, X2 = network.formula(x.trainLogit[-1], weight, 0.1, 0.2)
print(X2)
