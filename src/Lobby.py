import argparse, retro, os, time
from Discretizer import StreetFighter2Discretizer
from enum import Enum

class Lobby_Full_Exception(Exception):
    pass

class Lobby_Modes(Enum):
    SINGLE_PLAYER = 1
    TWO_PLAYER = 2

class Lobby():
    """ """

    ### Static Variables 

    # Variables relating to monitoring state and contorls
    NO_ACTION = [0] * 12
    ACTION_BUTTONS = ['X', 'Y', 'Z', 'A', 'B', 'C']
    ROUND_TIMER_NOT_STARTED = 39208
    STANDING_STATUS = 512
    CROUCHING_STATUS = 514
    JUMPING_STATUS = 516
    ACTIONABLE_STATUSES = [STANDING_STATUS, CROUCHING_STATUS, JUMPING_STATUS]

    FRAME_RATE = 1 / 115                                                                           # The time between frames if real time is enabled
    
    ### End of static variables 

    ### Static Methods

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

    def __init__(self, game= 'StreetFighterIISpecialChampionEdition-Genesis', render= False, mode= Lobby_Modes.SINGLE_PLAYER):
        """Initializes the agent and the underlying neural network

        Parameters
        ----------
        game
            A String of the game the lobby will be making an environment of, defaults to StreetFighterIISpecialChampionEdition-Genesis

        render
            A boolean flag that specifies whether or not to visually render the game while a match is being played

        Returns
        -------
        None
        """
        self.game = game
        self.render = render
        self.mode = mode
        self.clearLobby()

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
        self.environment = retro.make(game= self.game, state= state, players= self.mode.value)
        #self.environment = StreetFighter2Discretizer(self.environment)
        self.environment.reset()                                                               
        self.lastObservation, _, _, self.lastInfo = self.environment.step(Lobby.NO_ACTION)                   # The initial observation and state info are gathered by doing nothing the first frame and viewing the return data
        self.done = False
        while not self.isActionableState(self.lastInfo, Lobby.NO_ACTION):
            self.lastObservation, _, _, self.lastInfo = self.environment.step(Lobby.NO_ACTION)

    def addPlayer(self, newPlayer):
        for playerNum, player in enumerate(self.players):
            if player is None:
                self.players[playerNum] = newPlayer
                return

        raise Lobby_Full_Exception("Lobby has already reached the maximum number of players")

    def clearLobby(self):
        self.players = [None] * self.mode.value

    def isActionableState(self, info, action = 0):
        """Determines if the Agent has control over the game in it's current state(the Agent is in hit stun, ending lag, etc.)

        Parameters
        ----------
        action
            The last action taken by the Agent

        info
            The RAM info of the current game state the Agent is presented with as a dictionary of keyworded values from Data.json

        Returns
        -------
        isActionable
            A boolean variable describing whether the Agent has control over the given state of the game
        """
        action = self.environment.get_action_meaning(action)
        if info['round_timer'] == Lobby.ROUND_TIMER_NOT_STARTED:                                                       
            return False
        elif info['status'] == Lobby.JUMPING_STATUS and any([button in action for button in Lobby.ACTION_BUTTONS]):   # Have to manually track if we are in a jumping attack
             return False
        elif info['status'] not in Lobby.ACTIONABLE_STATUSES:                                                         # Standing, Crouching, or Jumping 
             return False
        else:
             return True

    def play(self, state, realTime= False):
        """The Agent will load the specified save state and play through it until finished, recording the fight for training

        Parameters
        ----------
        state
            A string of the name of the save state the Agent will be playing

        realTime
            A boolean flag used to slow the game down to approximately real game speed to make viewing for humans easier, defaults to false

        Returns
        -------
        None
        """
        self.initEnvironment(state)
        while not self.done:
            if self.render: self.environment.render()
            
            self.lastAction = self.players[0].getMove(self.environment.action_space, self.lastObservation, self.lastInfo)
            obs, self.lastReward, self.done, info = self.environment.step(self.lastAction)
            while not self.isActionableState(info, action = self.lastAction):
                obs, tempReward, self.done, info = self.environment.step(Lobby.NO_ACTION)
                if self.render: self.environment.render()
                if realTime: time.sleep(Lobby.FRAME_RATE)
                self.lastReward += tempReward

            self.players[0].recordStep(self.lastObservation, self.lastInfo, self.lastAction, self.lastReward, obs, info, self.done)
            self.lastObservation, self.lastInfo = [obs, info]                               # Overwrite after recording step so Agent remembers the previous state that led to this one
            if realTime: time.sleep(Lobby.FRAME_RATE)
        self.environment.close()
        self.environment.viewer.close()

    def executeTrainingRun(self, review= True, episodes= 1, realTime= False):
        """The lobby will load each of the saved states to generate data for the agent to train on
            Note: This will only work for single player mode

        Parameters
        ----------
        review
            A boolean variable that tells the Agent whether or not it should train after running through all the save states, true means train

        episodes
            An integer that represents the number of game play episodes to go through before training, once through the roster is one episode

        realTime
            A boolean flag used to slow the game down to approximately real game speed to make viewing for humans easier, defaults to false

        Returns
        -------
        None
        """
        for episodeNumber in range(episodes):
            print('Starting episode', episodeNumber)
            self.players[0].prepareForFight()                                       # Double ended queue that stores states during the game
            for state in Lobby.getStates():
                self.play(state= state, realTime= realTime)
            
            if self.players[0].__class__.__name__ != "Agent" and review == True: 
                self.players[0].reviewFight()

if __name__ == "__main__":
    testLobby = Lobby(render= True)
    from Agent import Agent
    agent = Agent()
    testLobby.addPlayer(agent)
    testLobby.executeTrainingRun(realTime= True)