from DeepQAgent import *

if __name__ == "__main__":
    qAgent = DeepQAgent(render= True, epsilon= 0.01)
    qAgent.load("../weights/StreetFighterWeights")
    qAgent.train(review= False, episodes= 1)
