from DeepQAgent import *

"""Makes a DeepQ Agent and runs it through one fight for each character in the roster so the user can view it"""
if __name__ == "__main__":
    qAgent = DeepQAgent(render= True, load= True)
    qAgent.train(review= False, episodes= 1)
