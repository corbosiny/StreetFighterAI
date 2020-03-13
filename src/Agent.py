import argparse, retro, threading, os

class Agent():

    def getStates():
        files = os.listdir('../StreetFighterIISpecialChampionEdition-Genesis')
        states = [file.split('.')[0] for file in files if file.split('.')[1] == 'state']
        return states

    def __init__(self, game= 'StreetFighterIISpecialChampionEdition-Genesis', render= True):
        self.game = game
        self.render = render
        self.stepHistory = []
        if self.__class__.__name__ != "Agent": self.initializeNetwork()

    def train(self, review= True, initialPopulation= False):
        for state in Agent.getStates():
            self.play(state= state, initialPopulation= initialPopulation)
        if self.__class__.__name__ != "Agent" and review == True: self.reviewGames()

    def play(self, state= 'chunli', initialPopulation= False):
        self.fighter = state
        self.initEnvironment(state)
        while not self.done:
            if initialPopulation: self.lastAction = self.getRandomMove()
            else: self.lastAction = self.getMove(self.lastObservation, self.lastInfo)

            obs, self.lastReward, self.done, self.lastInfo = self.environment.step(self.lastAction)
            self.recordStep()
            self.lastObservation = obs

            if self.render: self.environment.render()

        self.environment.close()

    def getMove(self, obs, info):
        return self.getRandomMove()

    def getRandomMove(self):
        return self.environment.action_space.sample()

    def recordStep(self):
        self.stepHistory.append([self.lastObservation, self.lastAction, self.lastReward, self.lastInfo, self.fighter])

    def initializeNetwork(self):
        raise NotImplementedError("Implement this is in the inherited agent")

    def initEnvironment(self, state):
        self.environment = retro.make(self.game, state)
        self.lastObservation = self.environment.reset() 
        self.stepHistory, self.lastInfo = [], []
        self.done = False

    def initialPopulation(self):
        self.train(initialPopulation= True)

    def reviewGames(self):
        self.prepareData()
        xTrain, xTest, yTrain, yTest = self.getTrainTestSplit()
        self.trainNetwork(xTrain, xTest, yTrain, yTest)

    def prepareData(self):
        raise NotImplementedError("Implement this is in the inherited agent")

    def getTrainTestSplit(self):
        return [None, None, None, None]

    def trainNetwork(self, xTrain, xTest, yTrain, yTest):
        raise NotImplementedError("Implement this is in the inherited agent")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processes agent parameters.')
    parser.add_argument('-r', '--render', action= 'store_true', help= 'Boolean flag for if the user wants the game environment to render during play')
    args = parser.parse_args()
    randomAgent = Agent(render= args.render)
    randomAgent.initialPopulation()