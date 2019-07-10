import os, sys
import numpy as np
import re

def grad(texty, QK, d, lambda1, my, X, PK1):
    XT = Transpose(X)
    QKT = Transpose(QK)
    PK1T = Transpose(PK1)

    idMatrix = []
    for a in range(d):
        idMatrix.append([])
        for b in range(d):
            if a == b:
                idMatrix[-1].append(1)
            else:
                idMatrix[-1].append(0)

    if texty == "P":
        print(idMatrix)
        return QK*XT/(QK*QKT + lambda1*idMatrix)

    else:
        return PK1*X/(PK1*PK1T + my*idMatrix)

def Transpose(listy):
    tempy = []
    for a in range(len(listy[0])):
        tempy.append([])
        for b in range(len(listy)):
            tempy[-1].append(listy[b][a])
    return tempy

class closedForm:
    def __init__(self):
        pass

    def P(self, Qk, QkT, lambda1, idMatrix, XT):
        return Qk*XT/(Qk*QkT + lambda1*idMatrix)

    def Q(self, PK1, X, PK1T, my, idMatrix):
        return PK1*X/(PK1*PK1T+my*idMatrix)

class GradientDescent:
    def __init__(self):
        pass

    def P(self, PK, lambda1, SUM, gradient, P, f, PKT, QKT):
        return PK - lambda1*SUM(gradient(P)*f(PKT, QKT))

    def Q(self, PK, lambda1, SUM, gradient, Q, f, PKT1, QKT):
        return PK - lambda1*SUM(gradient(Q)*f(PKT1, QKT))


class StochasticGradientDescent:
    def __init__(self):
        pass

    def P(self, PK, lambda1, gradient, QK, B):
        return PK - lambda1 * gradient(PK, QK, B)

    def Q(self, QK, lambda1, gradient, PK1, B):
        return QK - lambda1 * gradient(PK1, QK, B)