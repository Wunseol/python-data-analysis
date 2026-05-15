import nsfg
import thinkstats2
import thinkplot
import numpy as np

def Pumpkin(t):

    mu,var = thinkstats2.MeanVar(t)

    print ('均值:', mu)
    print ('方差:', var)
    print('标准差:', var ** 0.5)

pumpkins = [1,3,3,591]
Pumpkin( pumpkins )

# a = [1,3,3,591]
# b = [1,3,3,591]
# print(np.dot(a, b))
# print(np.var(a))