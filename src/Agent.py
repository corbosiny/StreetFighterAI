import argparse, retro, threading, os, numpy
from collections import deque

from tensorflow.python import keras
from keras.models import load_model

MAX_DATA_LENGTH = 50000

class Agent():
    """ Abstract class that user created Agents should inherit from.
        Contains helper functions for launching training environments and generating training data sets.
    """

    ### Static Variables 

    # The indicies representing what each index in a training point represent
    STATE_INDEX = 0                                                                    # The state the agent was presented with    
    ACTION_INDEX = 1                                                                   # The action the agent took
    REWARD_INDEX = 2                                                                   # The reward the agent received for that action
    NEXT_STATE_INDEX = 3                                                               # The next state that the action led to
    DONE_INDEX = 4                                                                     # A flag signifying if the game is over

    DEFAULT_MODELS_DIR_PATH = '../models'
    DEFAULT_LOGS_DIR_PATH = '../logs'
    
    ### End of static variables 


    ### Static Methods

    def getModelName(self):
        """Returns the formatted model name for the current model"""
        return  self.__class__.__name__ + "Model"

    def getLogsName(self):
        """Returns the formatted log name for the current model"""
        return self.__class__.__name__ + "logs"

    def getStates():
        """Static method that gets and returns a list of all the save state names that can be loaded

        Parameters
        ----------
        None

        Returns
        -------
        states
            A list of strings where each string is the name of a different save state
        """
        files = os.listdir('../StreetFighterIISpecialChampionEdition-Genesis')
        states = [file.split('.')[0] for file in files if file.split('.')[1] == 'state']
        return states

    ### End of static methods

    ### Object methods

    def __init__(self, game= 'StreetFighterIISpecialChampionEdition-Genesis', render= False, load= False):
        """Initializes the agent and the underlying neural network

        Parameters
        ----------
        game
            A String of the game the Agent will be making an environment of, defaults to StreetFighterIISpecialChampionEdition-Genesis

        render
            A boolean flag that specifies whether or not to visually render the game while the Agent is playing
        
        load
            A boolean flag that specifies whether to initialize the model from scratch or load in a pretrained model

        Returns
        -------
        None
        """
        self.game = game
        self.render = render
        self.memory = deque(maxlen= MAX_DATA_LENGTH)                                   # Double ended queue that stores states during the game
        if self.__class__.__name__ != "Agent":
            if load: 
                self.initializeNetwork()    								           # Only invoked in child subclasses, Agent has no network
            else: 
                self.loadModel()

    def train(self, review= True, episodes= 1):
        """Causes the Agent to run through each save state fight and record the results to review after

        Parameters
        ----------
        review
            A boolean variable that tells the Agent whether or not it should train after running through all the save states, true means train

        episodes
            An integer that represents the number of game play episodes to go through before training, once through the roster is one episode

        Returns
        -------
        None
        """
        for _ in range(episodes):
            print('Starting episode', _)
            self.memory = deque(maxlen= MAX_DATA_LENGTH)
            for state in Agent.getStates():
                self.play(state= state)
            if self.__class__.__name__ != "Agent" and review == True: 
                data = self.prepareMemoryForTraining(self.memory)
                self.trainNetwork(data)   		                                       # Only invoked in child subclasses, Agent does not learn

    def play(self, state):
        """The Agent will load the specified save state and play through it until finished, recording the fight for training

        Parameters
        ----------
        state
            A string of the name of the save state the Agent will be playing

        Returns
        -------
        None
        """
        self.initEnvironment(state)
        while not self.done:
            if self.render: self.environment.render()
            
            self.lastAction = self.getMove(self.lastObservation, self.lastInfo)
            obs, self.lastReward, self.done, info = self.environment.step(self.lastAction)

            self.recordStep(self.lastInfo, self.lastAction, self.lastReward, info, self.done)
            self.lastObservation, self.lastInfo = [obs, info]                          # Overwrite after recording step so Agent remembers the previous state that led to this one

        self.environment.close()

    def getRandomMove(self):
        """Returns a random set of button inputs

        Parameters
        ----------
        None

        Returns
        -------
            move a binary array of random button press combinations within the environments action space
        """
        move = self.environment.action_space.sample()                                  # Take random sample of all the button press inputs the Agent could make
        return move                                

    def recordStep(self, state, action, reward, nextState, done):
        """Records the last observation, action, reward and the resultant observation about the environment for later training
        Parameters
        ----------
        state
            The state the Agent was presented with before it took an action.
            A dictionary containing tagged RAM data

        action
            A multivariable array where each index represents a button press
            A one means the button was pressed, 0 means it was not

        reward
            The reward the agent received for taking that action

        nextState
            The state that the chosen action led to

        done
            Whether or not the new state marks the completion of the emulation

        Returns
        -------
        None
        """
        self.memory.append((state, action, reward, nextState, done))                     # Steps are stored as tuples to avoid unintended changes

    def initEnvironment(self, state):
        """Initializes a game environment that the Agent can play a save state in

        Parameters
        ----------
        state
            A string of the name of the save state to load into the environment

        Returns
        -------
        None
        """
        self.environment = retro.make(self.game, state)
        self.environment.reset() 
        firstAction = numpy.zeros(len(self.environment.action_space.sample()))           # The first action is always nothing in order for the Agent to get it's first set of infos before acting
        self.lastObservation, _, _, self.lastInfo = self.environment.step(firstAction)   # The initial observation and state info are gathered by doing nothing the first frame and viewing the return data
        self.done = False


    def loadModel(self):
        """Loads in pretrained model object ../models/{Agent_Class_Name}Model
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model = load_model(os.path.join(Agent.DEFAULT_MODELS_DIR_PATH, self.getModelName()))
        print("Model successfully loaded")

    def saveModel(self):
        """Saves the currently trained model in the default naming convention ../models/{Agent_Class_Name}Model
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model.save(os.path.join(Agent.DEFAULT_MODELS_DIR_PATH, self.getModelName()))
        print('Checkpoint established. model successfully saved')
        with open(os.path.join(Agent.DEFAULT_LOGS_DIR_PATH, self.getLogsName()), 'a+') as file:
            for loss in self.lossHistory.losses:
                file.write(str(loss))
                file.write('\n')

    ### End of object methods

    ### Abstract methods for the child Agent to implement
    def getMove(self, obs, info):
        """Returns a set of button inputs generated by the Agent's network after looking at the current observation

        Parameters
        ----------
        obs
            The observation of the current environment, 2D numpy array of pixel values

        info
            An array of information about the current environment, like player health, enemy health, matches won, and matches lost, etc.
            A full list of info can be found in data.json

        Returns
        -------
        move
            A set of button inputs in a multivariate array of the form Up, Down, Left, Right, A, B, X, Y, L, R.
        """
        return self.getRandomMove()

    def initializeNetwork(self):
        """To be implemented in child class, should initialize or load in the Agent's neural network"""
        raise NotImplementedError("Implement this is in the inherited agent")
    
    def prepareMemoryForTraining(self, memory):
        raise NotImplementedError("Implement this is in the inherited agent")

    def trainNetwork(self, data):
        """To be implemented in child class, Runs through a training epoch reviewing the training data
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError("Implement this is in the inherited agent")

    ### End of Abstract methods

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processes agent parameters.')
    parser.add_argument('-r', '--render', action= 'store_true', help= 'Boolean flag for if the user wants the game environment to render during play')
    args = parser.parse_args()
    randomAgent = Agent(render= args.render)
    randomAgent.train()
