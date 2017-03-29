'''
Created on Mar 26, 2017

@author: Xing Wang
'''

import numpy as np
from scipy.sparse import dok_matrix
import mdptoolbox
import copy
import time

PROBLEM = "Bandit"
INSTANCE_NUMBER = 1
INITIAL_STATE = [0,0,0,0]
# INSTANCE_NUMBER = 1
# INITIAL_STATE = [0,0,0,0]

ACTIONS = 4
STATES = 3**ACTIONS
HORIZON = 10

def convertIndexToList(stateIndex):
    return [int(x) for x in np.base_repr(stateIndex, 3, ACTIONS)[-ACTIONS::] ]

def convertListToIndex(stateList):
    return int("".join(str(x) for x in stateList), 3)
        
def getTransitionAndRewardMatrices():
    P = [dok_matrix((STATES, STATES)) for action in range(ACTIONS)]
    R = np.zeros((STATES, ACTIONS))
    for action in range(ACTIONS):
        for state in range(STATES):
            currentState = convertIndexToList(state)
            nextStates = reachableStates(currentState, action)
            for nextState in nextStates:
                sprime = convertListToIndex(nextState)
                P[action][state, sprime] = transitionProbability(currentState, action, nextState)
                #P[action][currentState, nextState] = transitionProbability(currentState, action, nextState)
                #print currentState, action, nextState, P[action][state, sprime]
            R[state, action] = immediateReward(currentState, action)
        P[action] = P[action].tocsr()
    return (P, R)

def transitionProbability(currentState, action, nextState):
    othersEqual = [currentState[i]==nextState[i] for i in range(ACTIONS) if i != action]
    if sum(othersEqual) < ACTIONS - 1:
        return 0.
    if currentState[action] == 0:
        if nextState[action] == 0:
            return 0.2
        elif nextState[action] == 1:
            return 0.8
    elif currentState[action] == 1:
        if nextState[action] == 1:
            return 0.5
        elif nextState[action] == 2:
            return 0.5
    elif currentState[action] == 2 and nextState[action] == 2:
        return 1.
    return 0.

def immediateReward(currentState, action):
    if (currentState[action] == 1):
        finishTaskReward = (action+1) ** 2
        finishTaskState = copy.copy(currentState)
        finishTaskState[action] = 2
        finishTaskProbability = transitionProbability(currentState, action, finishTaskState)
        expectedReward = finishTaskReward * finishTaskProbability * 1.
        return expectedReward
    return 0.

def reachableStates(stateTuple, action):
    ret = [stateTuple]
    if stateTuple[action] < 2:
        nextState = copy.copy(stateTuple)
        nextState[action] += 1
        ret.append(nextState)
    return ret

#print np.base_repr(80, base=3, padding=4)[-4::]
#print int("".join(str(x) for x in [2, 2, 2, 2]), 3)

if __name__ == "__main__":
    P, R = getTransitionAndRewardMatrices()
    startTime = time.time()
    fh = mdptoolbox.mdp.FiniteHorizon(P, R, 1, HORIZON)
    print P[0].shape
    fh.run()
    elaspedTime = time.time() - startTime
    print 'Running time = ', elaspedTime, ' seconds.'
    value = fh.V
    policy = fh.policy
    for s in range(policy.shape[0]):
#         print 'State: ', convertIndexToList(s), ' ------> Policy: ', policy[s,:]
#         print 'State: ', convertIndexToList(s), ' ------> Values: ', value[s,:]
        if s == convertListToIndex(INITIAL_STATE):
            optimalPolicy = str(policy[s, :])
            expectedReward = value[s, 0]
            
    f = open("sol_%s_%s_mdptoolbox.txt" % (PROBLEM, INSTANCE_NUMBER), 'w')
#     f.write('\t\t' +' ------> Values: ' + np.array2string(value[s,:], precision=4) + '\n')
#     f = open("sol_%s_%d.txt" % (PROBLEM, INSTANCE_NUMBER), 'w')
    f.write('Xing Wang\n')
    f.write("Problem: Bandit_%d\n" % INSTANCE_NUMBER)
    f.write("CPU Time: %.4f seconds\n" % elaspedTime)
    f.write("Expected Total Reward: %.4f \n" % expectedReward)
    f.write("Optimal Policy: %s \n" % optimalPolicy)
    #f.write('State: ' +  str(convertIndexToList(s)) + ' ------> Policy: ' + str(policy[s,:]) + '\n')
    for s in range(policy.shape[0]):
        f.write('State: ' + str(convertIndexToList(s)) + ' ------> Policy: ' + str(policy[s,:]) + '\n' )
    f.close()
    f.close()