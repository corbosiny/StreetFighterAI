import argparse, retro, threading, os

class Agent():
    """ Abstract class that user created Agents should inherit from.
        Contains helper functions for launching training environments and generating training data sets.
    """

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
        self.stepHistory = []
        if self.__class__.__name__ != "Agent": self.initializeNetwork()

    def train(self, review= True, initialPopulation= False):
        """Causes the Agent to run through each save state fight and record the results to review after

        Parameters
        ----------
        review
            A boolean variable that tells the Agent whether or not it should train after running through all the save states, true means train

        initialPopulation
            A boolean that lets the Agent know if it is generating the initial dataset to train on. If true it will make random moves just to generate an initial dataset

        Returns
        -------
        None
        """
        for state in Agent.getStates():
            self.play(state= state, initialPopulation= initialPopulation)
        if self.__class__.__name__ != "Agent" and review == True: self.reviewGames()

    def play(self, state= 'chunli', initialPopulation= False):
        """The Agent will load the specified save state and play through it until finished, recording the fight for training

        Parameters
        ----------
        state
            A string of the name of the save state the Agent will be playing

        initialPopulation
            A boolean that lets the Agent know if it is generating the initial dataset to train on. If true it will make random moves just to generate an initial dataset

        Returns
        -------
        None
        """
        self.fighter = state
        self.initEnvironment(state)
        while not self.done:
            if self.render: self.environment.render()

            if initialPopulation: self.lastAction = self.getRandomMove()
            else: self.lastAction = self.getMove(self.lastObservation, self.lastInfo)
            obs, self.lastReward, self.done, self.lastInfo = self.environment.step(self.lastAction)
            self.recordStep()
            self.lastObservation = obs

        self.environment.close()

    def getRandomMove(self):
        """Returns a random set of button inputs

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return self.environment.action_space.sample()

    def recordStep(self):
        """Records the last observation, action, reward and info about the environment along with the fighter name for training purposes
           Observation is a 2D array of all the pixels and their RGB color values of the current frame. Action is the multivariable array
           signifying the current button inputs. Reward is the reward value resultant of that action. And info is a list of predefined
           variables in ROM specified in data.json.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.stepHistory.append([self.lastObservation, self.lastAction, self.lastReward, self.lastInfo, self.fighter])

    def initEnvironment(self, state):
        """Initializes a game environment that the Agent can play in

        Parameters
        ----------
        state
            A string of the name of the save state to load into the environment

        Returns
        -------
        None
        """
        self.environment = retro.make(self.game, state)
        self.lastObservation = self.environment.reset() 
        self.stepHistory, self.lastInfo = [], []
        self.done = False

    def initialPopulation(self):
        """Initializes an intial dataset of the Agent playing randomly to train on and begins the first epoch of training
        Parameters
        ----------
        state
            The name of the save state to load into the environment

        Returns
        -------
        None
        """
        self.train(initialPopulation= True)

    def reviewGames(self):
        """Prepares the data and then runs through a training epoch reviewing each fight frame by frame
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.prepareData()
        xTrain, xTest, yTrain, yTest = self.getTrainTestSplit()
        self.trainNetwork(xTrain, xTest, yTrain, yTest)

    def getMove(self, obs, info):
        """Returns a set of button inputs generated by the Agent's network after looking at the current observation

        Parameters
        ----------
        obs
            The observation of the current environment, 2D numpy array of pixel values

        info
            An array of information about the current environment, like player health, enemy health, matches won, and matches lost

        Returns
        -------
        move
            A set of button inputs in a multivariate array of the form Up, Down, Left, Right, A, B, X, Y, L, R.
        """
        return self.getRandomMove()

    def initializeNetwork(self):
        """To be implemented in child class, should initialize or load in the Agent's neural network"""
        raise NotImplementedError("Implement this is in the inherited agent")

    def prepareData(self):
        """To be implemented in child class, prepares the data stored in self.stepHistory in anyway needed for training, can just be pass if unecessary
            The data is stored in self.recordStep and the formatting can be seen there.
        """
        raise NotImplementedError("Implement this is in the inherited agent")

    def getTrainTestSplit(self):
        """Splits the data into a train and test set
        Parameters
        ----------
        None

        Returns
        -------
            train_x, test_x, train_y, test_y which are all arrays of training points
        """
        raise NotImplementedError("Implement this is in the inherited agent")

    def trainNetwork(self, xTrain, xTest, yTrain, yTest):
        """To be implemented in child class, Runs through a training epoch reviewing the training data
        Parameters
        ----------
            train_x, test_x, train_y, test_y which are all areas of training points

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
