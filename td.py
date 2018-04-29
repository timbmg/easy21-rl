from environment import Easy21
import utils
import numpy as np
import dill as pickle

env = Easy21()
N0 = 100
actions = [0, 1]

def reset():
    Q = np.zeros((22, 11, len(actions))) # state-action value
    NSA = np.zeros((22, 11, len(actions))) # state-action counter
    wins = 0

    return Q, NSA, wins

Q, NSA, wins = reset()
trueQ = pickle.load(open('Q.dill', 'rb'))

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

episodes = int(1e4)
lmds = list(np.arange(0,11)/10)

mselambdas = np.zeros((len(lmds), episodes))
finalMSE = np.zeros(len(lmds))

for li, lmd in enumerate(lmds):

    Q, NSA, wins = reset()

    for episode in range(episodes):

        terminated = False
        E = np.zeros((22, 11, len(actions))) # Eligibility Trace
        p, d = env.initGame()
        # inital state and first action
        a = epsilonGreedy(p, d)
        SA = list()

        # Sample Environment
        while not terminated:

            pPrime, dPrime, r, terminated = env.step(p, d, a)

            if not terminated:
                aPrime = epsilonGreedy(pPrime, dPrime)
                tdError = r + Q[pPrime, dPrime, aPrime] - Q[p, d, a]
            else:
                tdError = r - Q[p, d, a]

            E[p, d, a] += 1
            NSA[p, d, a] += 1
            SA.append([p, d, a])

            for (_p, _d, _a) in SA:
                Q[_p, _d, _a] += alpha(_p, _d, _a) * tdError * E[_p, _d, _a]
                E[_p, _d, _a] *= lmd

            if not terminated:
                p, d, a = pPrime, dPrime, aPrime

        # bookkeeping
        if r == 1:
            wins += 1

        mse = np.sum(np.square(Q-trueQ)) / (21*10*2)

        mselambdas[li, episode] = mse

        if episode % 1000 == 0 or episode+1==episodes:
            print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f"%(lmd, episode, mse, wins/(episode+1)))

    finalMSE[li] = mse

    print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f"%(lmd, episode, mse, wins/(episode+1)))
    print("--------")

utils.plotMseLambdas(finalMSE, lmds)
utils.plotMseEpisodesLambdas(mselambdas)
