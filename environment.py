import numpy as np

class Easy21():

    def __init__(self):
        self.minCardValue, self.maxCardValue = 1, 10
        self.dealerUpperBound = 17
        self.gameLowerBound, self.gameUpperBound = 0, 21

    @classmethod
    def actionSpace(self):
        return (0, 1)

    def initGame(self):
        return (np.random.randint(self.minCardValue, self.maxCardValue+1),
               np.random.randint(self.minCardValue, self.maxCardValue+1))

    def draw(self):
        value = np.random.randint(self.minCardValue, self.maxCardValue+1)

        if np.random.random() <= 1/3:
            return -value
        else:
            return value

    def step(self, playerValue, dealerValue, action):

        assert action in [0, 1], "Expection action in [0, 1] but got %i"%action

        if action == 0:

            playerValue += self.draw()

            # check if player busted
            if not (self.gameLowerBound < playerValue <= self.gameUpperBound):
                reward = -1
                terminated = True

            else:
                reward = 0
                terminated = False

        elif action == 1:
            terminated = True

            while self.gameLowerBound < dealerValue < self.dealerUpperBound:
                dealerValue += self.draw()

            # check if dealer busted // playerValue greater than dealerValue
            if not (self.gameLowerBound < dealerValue <= self.gameUpperBound) \
                or playerValue > dealerValue:
                reward = 1

            elif playerValue == dealerValue:
                reward = 0

            elif playerValue < dealerValue:
                reward = -1

        return playerValue, dealerValue, reward, terminated
