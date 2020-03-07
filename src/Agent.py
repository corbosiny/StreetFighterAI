import argparse, retro, threading, os

class Agent():

    def getStates():
        files = os.listdir('../StreetFighterIISpecialChampionEdition-Genesis')
        states = [file.split('.')[0] for file in files if file.split('.')[1] == 'state']
        return states

    def __init__(self, game= 'StreetFighterIISpecialChampionEdition-Genesis', render= True):
        self.game = game
        self.render = True
        self.stepHistory = []
        self.initializeNetwork()

    def train(self):
        for state in Agent.getStates():
            self.play(state= state)
        self.reviewGames()

    def play(self, state= 'chunli'):
        self.fighter = state
        self.environment = retro.make(self.game, state)
        self.lastObservation, self.stepHistory = self.environment.reset(), []
        while True:
            self.lastAction = self.getMove()
            obs, self.lastReward, done, self.lastInfo = self.environment.step(self.lastAction)
            self.recordStep()
            self.lastObservation = obs
            if self.render: self.environment.render()
            if done or self.lastInfo['matches_won'] == 2: break

        self.environment.close()

    def getMove(self):
        return self.environment.action_space.sample()

    def recordStep(self):
        self.stepHistory.append([self.lastObservation, self.lastAction, self.lastReward, self.lastInfo, self.fighter])

    def initializeNetwork(self):
        pass

    def reviewGames(self):
        self.prepareData()
        xTrain, xTest, yTrain, yTest = self.getTrainTestSplit()
        self.trainNetwork(xTrain, xTest, yTrain, yTest)

    def prepareData(self):
        pass

    def getTrainTestSplit(self):
        return [None, None, None, None]

    def trainNetwork(self, xTrain, xTest, yTrain, yTest):
        pass

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process agent parameterss.')
    
    randomAgent = Agent()
    randomAgent.train()