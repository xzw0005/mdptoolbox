'''
Created on Mar 26, 2017

@author: XING
'''

import numpy as np
import mdptoolbox
from scipy.sparse import dok_matrix
from scipy.stats import randint
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import geom
import time

PROBLEM = "QueueControl"

class QueueControl(object):
    '''
    classdocs
    '''

    def __init__(self, N=10, K=100, R1 = 2.0, R2 = 1.3, opt='a'):
        '''
        Constructor
        '''
        self.N = N
        self.K = K
        self.R1 = R1
        self.R2 = R2
        self.g = []
        self.f1=[]
        self.f2 = []
        self.F1bar = []
        self.F2bar = [];
        self.initProbs(opt)

    def actionsLookupDictionary(self):
        getIndexByAction = {}
        getActionByIndex = {}
        allPossibleActions = self.feasibleActions(self.K)
        self.numA = len(allPossibleActions)
        for i in range(len(allPossibleActions)):
            getActionByIndex[i] = allPossibleActions[i]
            getIndexByAction[tuple(allPossibleActions[i])] = i
        self.getIndexByAction = getIndexByAction
        self.getActionByIndex = getActionByIndex
#        return getActionByIndex, getIndexByAction
        
    def initProbs(self, opt):
        unif100 = randint(0, 101)
        pois50 = poisson(50)
        pois10 = poisson(10)
        binom100 = binom(100, 0.5)
        geom02 = geom(0.02)
        for k in range(self.K + 1):
            if (opt == 'a') or (opt == 'b'):
                self.g.append(unif100.pmf(k))
            elif (opt == 'c') or (opt == 'd'):
                self.g.append(binom100.pmf(k))
            if (opt == 'a') or (opt == 'c'):
                self.f1.append(pois50.pmf(k))
                self.f2.append(pois10.pmf(k))
                self.F1bar.append(1 - pois50.cdf(k))
                self.F2bar.append(1 - pois10.cdf(k))
            elif (opt == 'b') or (opt == 'd'):
                self.f1.append(geom02.pmf(k+1))
                self.f2.append(geom02.pmf(k+1))
                self.F1bar.append(1 - geom02.cdf(k+1))
                self.F2bar.append(1 - geom02.cdf(k+1))
        #print self.g[100]
                
    def immediateReward(self, action):
        [a1, a2] = action   # a1 is the number allowed to enter the 1st service queue, a2 is the number to the 2nd service queue 
        r = 0.
        for k in range(a1+1):
            r += self.R1 * self.f1[k] * k
        for k in range(a2+1):
            r += self.R2 * self.f2[k] * k
        r += self.R1 * self.F1bar[a1] * a1
        r += self.R2 * self.F2bar[a2] * a2
        return r - a1 - a2
    
    def transitionProbability(self, sprime):
        return self.g[sprime]
    
    def getTransitionMatrix(self):
#         #P = [dok_matrix((self.K+1, self.K+1)) for a in range(self.numA)]
#         pa = dok_matrix((self.K+1, self.K+1))
#         rowsum = 0
#         for sprime in range(self.K+1):
#             rowsum += self.g[sprime]
#             for s in range(self.K+1):
#                 pa[s, sprime] = self.g[sprime]
# #         print pa.sum(axis=1)
# #        print pa.todense()[self.K, :]
#         P = [pa.tocsr()] * self.numA
#         return P        
        
        #P = [dok_matrix((self.K+1, self.K+1)) for a in range(self.numA)]
        pa = dok_matrix((self.K+1, self.K+1))
        rowsum = 0
        for sprime in range(self.K+1):
            rowsum += self.g[sprime]
            for s in range(self.K+1):
                pa[s, sprime] = self.g[sprime]
#         print pa.sum(axis=1)
        for s in range(self.K+1):
            pa[s, self.K/2] += (1. - rowsum)
        P = [pa.tocsr()] * self.numA
        return P
        
    def getRewardMatrix(self):
        numActions = len(self.getActionByIndex)
        R = np.zeros((self.K + 1, numActions))
        for s in range(self.K + 1):
            actionList = self.feasibleActions(s)
            for a in actionList:
                reward = self.immediateReward(a)
                index = self.getIndexByAction[tuple(a)]
                R[s, index] = reward
#         print R[100, :]
        return R
        
    def feasibleActions(self, state):
        actionList = []
        for a1 in range(state + 1):
            for a2 in range(state+1 - a1):
                actionList.append([a1, a2])
        return actionList
    
    def getmdp(self):
        self.actionsLookupDictionary()
        P = self.getTransitionMatrix()
        R = self.getRewardMatrix()
        return P, R
    
    def solveMDP(self):
        P, R = self.getmdp()
        fh = mdptoolbox.mdp.FiniteHorizon(P, R, .95, self.N)
        fh.run()
        return fh

if __name__ == "__main__":
    OPT = 'd'
    instance = QueueControl(N=10, K=100, R1 = 10, R2 = 30, opt=OPT)
#    instance.getmdp()
    startTime = time.time()
    res = instance.solveMDP()
    elaspedTime = time.time() - startTime

    print 'Running time = ', elaspedTime, ' seconds.'
    
    policy = res.policy
    value = res.V
    for s in range(len(policy)):
        dtList = list(policy[s])
        dtList = [instance.getActionByIndex[dt] for dt in dtList]
#         print 'State: ', s, ' ------> ', dtList
#         print value[s]
#         f.write('State: ' +  str(s) + ' ------> Policy: ' + str(dtList) + '\n')
#         f.write('\t' +' ------> Values: ' + np.array2string(value[s,:], precision=4) + '\n')
        if s == 50:
            optimalPolicy = str(dtList)
            expectedReward = value[s, 0]
    print expectedReward
#     f.write('State: ' +  str(convertIndexToList(s)) + ' ------> Policy: ' + str(policy[s,:]) + '\n')
#     f.write('\t\t' +' ------> Values: ' + np.array2string(value[s,:], precision=4) + '\n')
        
    f = open("sol_%s_%s_discount_mdptoolbox.txt" % (PROBLEM, OPT), 'w')
    f.write('Xing Wang\n')
    f.write("Problem: %s_%s\n" % (PROBLEM, OPT))
    f.write("CPU Time: %.4f seconds\n" % elaspedTime)
    f.write("Expected Total Reward: %.4f \n" % expectedReward)
    f.write("Optimal Policy: \n ")
    for s in range(len(policy)):
        dtList = list(policy[s])
        dtList = [instance.getActionByIndex[dt] for dt in dtList]
        f.write('State: '+ str(s) + ' ------> '+ str(dtList) +'\n' )
    f.close()
 
#     print instance.getTransitionMatrix().tocsr().sum(axis=1)
#     print instance.getRewardMatrix().shape
    