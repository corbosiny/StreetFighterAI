import argparse, retro, threading, os, numpy
from collections import deque

MAX_DATA_LENGTH = 20000

class Agent():
    """ Abstract class that user created Agents should inherit from.
        Contains helper functions for launching training environments and generating training data sets.
    """

    # Static Variables for indexing into a point in the training data to find specific information
    STATE_INDEX = 0
    ACTION_INDEX = 1
    REWARD_INDEX = 2
    NEXT_STATE_INDEX = 3
    DONE_INDEX = 4

    doneKeys = [1024, 1026, 1028]

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

    def __init__(self, game= 'StreetFighterIISpecialChampionEdition-Genesis', render= False):
        """Initializes the agent and the underlying neural network

        Parameters
        ----------
        game
            A String of the game the Agent will be making an environment of, defaults to StreetFighterIISpecialChampionEdition-Genesis

        render
            A boolean that specifies whether or not to visually render the game while the Agent is playing

        Returns
        -------
        None
        """
        self.game = game
        self.render = render
        self.memory = deque(maxlen= MAX_DATA_LENGTH)                                    # Double ended queue that stores states during the game
        if self.__class__.__name__ != "Agent": self.initializeNetwork()                 # Only invoked in child subclasses, Agent has no network

    def train(self, review= True, episodes= 1):
        """Causes the Agent to run through each save state fight and record the results to review after

        Parameters
        ----------
        review
            A boolean variable that tells the Agent whether or not it should train after running through all the save states, true means train

        Returns
        -------
        None
        """
        for _ in range(episodes):
            self.memory = deque(maxlen= MAX_DATA_LENGTH)
            for state in Agent.getStates():
                self.play(state= state)
            if self.__class__.__name__ != "Agent" and review == True: self.trainNetwork()   # Only invoked in child subclasses, Agent does not learn

    def play(self, state= 'chunli'):
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
        """Records the last observation, action, reward and the resultant observation about the environment
           The states are a dict of predefined variables in RAM specified in data.json that describe the game.
           State is the previous game state before the action while nextState is the game state after the action. 
           Action is the multivariable array signifying the current button inputs, 1 means pressed and 0 is not.
           Reward is the resultant reward measured based on the success of the last action taken by the Agent.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if nextState['status'] in Agent.doneKeys or nextState['enemy_status'] in Agent.doneKeys: 
            return
        else:
            self.memory.append((self.prepareNetworkInputs(state), action, reward, self.prepareNetworkInputs(nextState), done))                    # Steps are stored as tuples to avoid unintended changes

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
        firstAction = numpy.zeros(len(self.environment.action_space.sample()))         # The first action is always nothing in order for the Agent to get it's first set of infos before acting
        self.lastObservation, _, _, self.lastInfo = self.environment.step(firstAction) # The initial observation and state info are gathered by doing nothing the first frame and viewing the return data
        self.done = False

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

    def prepareNetworkInputs(self, step):
        """To be implemented in child class, encodes a feature vector from the given state data
        
        Parameters
        ----------
        step
            A given set of state information from the environment
            
        Returns
        -------
        features
            A feature vector extracted from the step that is the same size as the network input layer
        """
        return step

    def trainNetwork(self):
        """To be implemented in child class, Runs through a training epoch reviewing the training data
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError("Implement this is in the inherited agent")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processes agent parameters.')
    parser.add_argument('-r', '--render', action= 'store_true', help= 'Boolean flag for if the user wants the game environment to render during play')
    args = parser.parse_args()
    randomAgent = Agent(render= args.render)
    randomAgent.train()
