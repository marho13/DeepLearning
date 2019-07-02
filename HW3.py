import re, sys, os, math, random

random.seed(42)

#f(x) = L2((W2W1Xi)-Yi) + lambda1(W1)frob + lambda2(W2)frob
#derivative = [2W2loss + 2lambda1, 2W1loss + 2lambda2] where the comma separates w1 and w2s formula (gradient)
#   also derivative is also not a pure derivative but instead a partial derivative with respect to w1, w2
#l2 norm = sum( (yi-( f(xi))**2 ) for all i
#frob norm = sqrt(sum i( sum j(aij**2)))

def l2norm(input):
    sum = 0
    for inp in input:
        for i in inp:
            sum += i**2
    return math.sqrt(sum)

def frobNorm(input):
    sum = 0
    for inp in input:
        for i in inp:
            sum += i ** 2
    return math.sqrt(sum)

class fileRead:
    def __init__(self, file, testRatio):
        #Variable Initiation
        self.ratio = testRatio
        self.fileName = file

        #Read data into dataset list, then find length of data
        self.dataset, self.header = self.read()
        self.length = len(self.dataset)        #Reads the file, then makes a list for each new line
        data = open(self.fileName, "r").read()
        data = data.split("\n")


        #Shuffle said list
        self.shuffleData()


    def read(self):

        header = data.pop(0).split(",")
        #Creates the header, and dataset
        dataset = [d.split(",") for d in data[1:]]

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
        w1 = w1 - W1Change*self.lr
        w2 = w2 - W1Change*self.lr
        return w1, w2



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
    def __init__(self, config):
        self.networkconfig = config
        self.weights = self.weightsInit()
        # self.bias = self.biasInit() #this does not seem to be in the optimization problem
        # self.model = self.networkCreation()

    def weightsInit(self):
        weights = []
        for a in range(len(self.networkconfig)): #15 values, predict 1 of them (bodyfat)
            weights.append([])
            for c in range(self.networkconfig[a][1]):
                weights[-1].append([])
                for d in range(self.networkconfig[a][0]):
                    weights[-1][-1].append(random.random())
        #
        # for w in range(len(weights)):
        #     for a in range(len(weights[w])):
        #         for b in range(len(weights[w][a])):
        #             weights[w][a][b] = random.random()



        #when the network is created, associate weights with each "neuron"
        return weights

    # def biasInit(self): #should we use bias, or just lambda1 and lambda2?
    #     #CHECK THIS!!!!
    #
    #     bias = []
    #     for a in range(len(self.networkconfig)):
    #         bias.append([])
    #         for b in range(len(self.networkconfig[a])):
    #             bias[-1].append(random.random())
    #     #same as weightsInit, but for bias
    #     return bias

    # def networkCreation(self):
    #     network = []
    #     for a in range(len(self.networkconfig)):
    #         network.append([])
    #         for b in range(len(self.networkconfig[a])):
    #             network[-1].append([self.weights[a][b], self.bias[a][b]])
    #     return network

    def fit(self, backprop, input):
        pass
        #Fit send the data in to the network, and do the activation and such
        #then use the backpropagation method, and so on

    def predict(self, input):
        pass
        #Like with fit, without any form of backprop

    def forwardpass(self, input, l2norm, ans):
        pass #take the input, and pass it through the weights of each layer
        #call l2norm the tempOut
        #call frob weights0 for bias1, and frob weights1 for bias2
        #to make x2 the same shape as ans, we need to do global average pooling or something like it!!!
        #if no globAvgPool is done, then the w2 shape needs to be the same as the output [1, 1]
        output = l2norm(self.weights[1]*self.weights[0]*input - ans) + lambda1*frobNorm(self.weights[0]) + self.lambda2*frobNorm(self.weights[1])

#Create an instance of the fileRead class
x = fileRead("hw3 - dataset/Bodyfat.csv", 0.2)

#Read the original file, and create the header and a dataset of said file
file, header = x.read()

#Split the data into testing/training labels and logits
train, test = x.splitTrainandTest()
trainLogit, trainLabel = x.labelLogit(train)
testLogit, testLabel = x.labelLogit(test)

network = LinearNetwork([[15, 2],[2, 15]])
weight = network.weights

