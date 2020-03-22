from DeepQAgent import *

if __name__ == "__main__":
    qAgent = DeepQAgent(render= True, load= True)
    qAgent.train(review= False, episodes= 1)
