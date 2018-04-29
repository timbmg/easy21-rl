from environment import Easy21
import utils

import numpy as np
import dill as pickle

printEvery = 10000
actions = [0, 1]

N0 = 100

# state, action function
Q = np.zeros((22, 11, len(actions)))

# number of times action a has been choosen in state s
NSA = np.zeros((22, 11, len(actions)))

# number of times state s has been visited
NS = lambda p, d: np.sum(NSA[p, d])

# step size
alpha = lambda p, d, a: 1/NSA[p, d, a]

# exploration probability
epsilon = lambda p, d: N0 / (N0 + NS(p, d))

def epsilonGreedy(p, d):
    if np.random.random() < epsilon(p, d):
        # explore
        action = np.random.choice(actions)

    else:
        # exploit
        action = np.argmax( [Q[p, d, a] for a in actions] )

    return action

episodes = int(1e7)
meanReturn = 0
wins = 0

env = Easy21()

for episode in range(episodes):

    terminated = False
    SAR = list() # state, action, reward
    p, d = env.initGame()
    # Sample Environment
    while not terminated:

        a = epsilonGreedy(p, d)

        NSA[p, d, a] += 1

        pPrime, dPrime, r, terminated = env.step(p, d, a)

        SAR.append([p, d, a, r])

        p, d = pPrime, dPrime

    # Update Q
    G = sum([sar[-1] for sar in SAR]) # sum all rewards
    for (p, d, a, _) in SAR:
        Q[p, d, a] += alpha(p, d, a) * (G - Q[p, d, a])

    # bookkeeping
    meanReturn = meanReturn + 1/(episode+1) * (G - meanReturn)
    if r == 1:
        wins += 1

    if episode % printEvery == 0:
        print("Episode %i, Mean-Return %.3f, Wins %.2f"%(episode, meanReturn, wins/(episode+1)))

pickle.dump(Q, open('Q.dill', 'wb'))
_ = pickle.load(open('Q.dill', 'rb')) # sanity check

utils.plot(Q, [0,1])
