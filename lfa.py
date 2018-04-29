from environment import Easy21
import utils
import numpy as np
import dill as pickle
import time

env = Easy21()
N0 = 100
actions = [0, 1]

def reset():
    theta = np.random.randn(3*6*2, 1)
    wins = 0

    return theta, wins

trueQ = pickle.load(open('Q.dill', 'rb'))

# step size
alpha = 0.01

# exploration probability
epsilon = 0.05

episodes = int(1e4)
lmds = list(np.arange(0,11)/10)

mselambdas = np.zeros((len(lmds), episodes))
finalMSE = np.zeros(len(lmds))

def epsilonGreedy(p, d):
    if np.random.random() < epsilon:
        # explore
        action = np.random.choice(actions)

    else:
        # exploit
        action = np.argmax( [Q(p, d, a) for a in actions] )

    return action

def features(p, d, a):
    f = np.zeros(3*6*2)

    for fi, (lower, upper) in enumerate(zip(range(1,8,3), range(4, 11, 3))):
        f[fi] = (lower <= d <= upper)

    for fi, (lower, upper) in enumerate(zip(range(1,17,3), range(6, 22, 3)), start=3):
        f[fi] = (lower <= p <= upper)

    f[-2] = 1 if a == 0 else 0
    f[-1] = 1 if a == 1 else 0

    return f.reshape(1, -1)

def Q(p, d, a):
    return np.dot(features(p,d,a), theta)

allFeatures = np.zeros((22, 11, 2, 3*6*2))
for p in range(1, 22):
    for d in range(1, 11):
        for a in range(0, 2):
            allFeatures[p-1, d-1, a] = features(p, d, a)

def allQ():
    return np.dot(allFeatures.reshape(-1, 3*6*2), theta).reshape(-1)


for li, lmd in enumerate(lmds):

    theta, wins = reset()

    for episode in range(episodes):

        terminated = False
        E = np.zeros_like(theta) # Eligibility Trace

        # inital state and first action
        p, d = env.initGame()
        a = epsilonGreedy(p, d)

        # Sample Environment
        while not terminated:

            pPrime, dPrime, r, terminated = env.step(p, d, a)

            if not terminated:
                aPrime = epsilonGreedy(pPrime, dPrime)
                tdError = r + Q(pPrime, dPrime, aPrime) - Q(p, d, a)
            else:
                tdError = r - Q(p, d, a)

            E = lmd * E + features(p, d, a).reshape(-1, 1)
            gradient = alpha * tdError * E
            theta = theta + gradient

            if not terminated:
                p, d, a = pPrime, dPrime, aPrime

        # bookkeeping
        if r == 1:
            wins += 1

        mse = np.sum(np.square(allQ() - trueQ.ravel())) / (21*10*2)
        mselambdas[li, episode] = mse

        if episode % 1000 == 0 or episode+1==episodes:
            print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f"%(lmd, episode, mse, wins/(episode+1)))

    finalMSE[li] = mse
    print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f"%(lmd, episode, mse, wins/(episode+1)))
    print("--------")

utils.plotMseLambdas(finalMSE, lmds)
utils.plotMseEpisodesLambdas(mselambdas)
