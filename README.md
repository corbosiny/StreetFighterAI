# Getting Started

This readme will take you through how to get this repo up and running with the example agents, make your own test agents, and also how to create your own save states to test your agents on. 

---
## Installing Dependancies

This code only works with Python 3.6 or later. Before trying to install dependancies it is recommend to open the terminal and run:  
`sudo apt-get update`  
`sudo apt-get upgrade`  
`sudo -H pip3 install --upgrade pip`  
To download the necessary dependancies after cloning the repo call:
`pip3 install -r requirements.txt`
This should be called in the top level directory of the repo. This will install the following libraries you will need to create game environments that serve as a wrapper abstracting the interface between your agent and the underlying emulator: 

**-gym**  
**-gym-retro**   
**-retro**  

These libraries can sometimes have serious issues installing themselves or their dependancies on a windows machine. It is recommended to work on Linux. The server we will be training on runs Linux and all libraries plus code have been confirmed to work on Ubuntu's latest stable distribution.

---
## Preparing the Game Files 

After the dependancies have been installed the necessary game files, all zipped inside of the **StreetFighterIISpecialChampionEdition-Genesis** directory, can be setup. The game files need to be copied into the actual game data files inside the installation of the retro library on your local machine. This location can be found by running the following lines in the command line:  

`python3`  
`import retro`  
`print(retro.__file__)`    

That should return the path to where the retro __init__.py script is stored. One level up from that should be the data folder. Inside there should be the stable folder. Copy the **StreetFighterIISpecialChampionEdition-Genesis.zip** here. There is already an existing folder here of the same name but we are going to overwrite that so tell your computer to replace the existing files when the prompt comes up. Inside the folder should be the following files:

-rom.md    
-rom.sha    
-scenario.json  
-data.json  
-metadata.json  
-Several .state files with each having the name of a specifc fighter from the game  

With that the game files should be correctly set up and you should be able to run a test agent. 

---
## The first test run

To double check that the game files were properly set up the example agent can be run. cd into the src directory. Then either run the following command on your terminal:

`python3 Agent.py`

Or you can open Agent.py and excute it from your prefered IDE of choice. This is an Agent that essentially button mashes. It does a random move every frame update despite what is going on in the game. If everything was installed correctly it should simply one by one open up each save state in the game directory and run through the fight set up for it. This will involve a small window popping up showing the game running at a very high speed. Once the fight is over a new window should open up with the next fight. Once all fights are over the program should kill itself and close all windows. 

---
## How to make an agent

To make your own agent it is recommended to make a class that inherits from Agent.py. Agent.py contains several useful helper functions that allow you to:

-Find and loop through every save state  
-Handle opening and cleaning up of the gym environment  
-Recording data during the fights
-And even training the agent after the fights

Some of these methods aren't filled in but the overall framework in place makes it easy to drop in your own versions of the getMove, train, and initialize network functions such that there is little work outside of network design that has to be done to get your agent up and running. The goal is to create a streamlined platform to rapidly prototype, train, and deploy new agents instead of starting for scratch everytime. As well enforcing the interface for the agent class allows for high level software to be developed that can import various user created agents without fear of breaking due to interface issues. 

### Agent class

There are four main functions that need to be implemented in order to create a new intelligent agent.

-getMove  
-initializeNetwork  
-prepareData  
-trainNetwork

#### getMove

Get move simply returns a multivariate array that has ones in the input slots signifying which buttons the agent is pressing for the next frame update. 

#### initializeNetwork

initializeNetwork does whatever under the hood set up needs to be done to either create a network from scratch or load in an existing network and weights

#### prepareData

preps the recorded data of each match into such a way that it can be fed properly into the structure of the network this agent is using

#### trainNetwork

runs a training epoch on the preparedData from all the last round of recorded fights

## Jason Files

There are three jason files that the gym environment reads in order to setup the high level "rules" of the emulation. These files are metadata.json, data.json, and scenario.json. 

### Metadata.json

The metadata.json file holds high level global information about the game environment. For now this simply tells the environment the default save state that the game ROM should launch in if none has been selected. 

### Data.json

The data.json file is an abstraction of the games ram into callable variables with specified data types that the environment, user, and environment.json files can interact with. For now it specifies the memory addresses where the count down timer, agent round win counter, enemy round win counter, score, agent health, and enemy health can be found. 

### Scenario.json

Scenario.json specifies several conditions over which that define the goal of the simulation or specify what criteria the agent will be judged on for rewards. The two main specifications are the reward function and the done flag.

#### Reward Function

The reward functione specifies what variables make up the reward function and what weights are assigned, whether that be positive or negative, to each variable. After each action is taken by an agent a reward calculated by this function is returned to the agent. This is then recorded and stored for later training after all fights in an epoch are finished. For now the default reward function utilizes the agent's health, the enemy health, the number of rounds the agent has won, and the number of rounds the enemy has won. 

#### Done

Done is a flag that signifies whether the current environment has completed. Currently Done is set if the enemey or the agent get two round wins, which in game is what determines if a match is over. So once the match is over the agent moves onto the next save state.

---
## Generating New Save States

Save states are generated by a user actually saving their games state while playing in an emulator. In order to make new save states to contribute to the variety of matches your agent will play in you have to actually play the Street Fighter ROM up until the point you want the agent to start at. 

### Installing the Emulator

Retroarch is the emulator that is needed to generate the correct save states under the hood. It can be installed at:  
https://www.retroarch.com/?page=platforms


### Preparing the Cores

Retroarch needs a core of the architecture it is trying to simulate. The Street Fighter ROM we are working with is for the Sega Genisis. Retro actually has a built in core that can be copy and pasted into Retroarchs core folder and this is their recommended installation method. However finding the retroarch installation folder can be difficult and so can finding the cores in the Retro library. Instead open up Retroarch and go into Load Core. Inside Load Core scroll down and select download core. Scroll way down until you see genesis_plus_gx_libretro.so.zip and install it. Now go back to the main menu and select Load Content. Navigate to the Street Fighter folder at the top level of the repo and load the rom.md file. From here the game should load up correctly.

### Saving states

F2 is the shortcut key that saves the current state of the game. The state is saved to the currently selected game state slot. This starts at slot zero and can be incremented with the F6 key and decremented with the F7 key. When a fight is about to start that you want to create a state for hit F2. Then I would recommend incrementing the save slot by pressing F6 so that if you try to save another state you don't accidentally overwrite the last state you saved. There are 8 slots in total. By pressing F5 and going to view->settings-Directory you can control where the save states are stored. The states will be saved with the extension of 'state' plus the number of the save slot it was saved in. To prep these for usage cleave off the number at the end of each extension and rename each file to the name of the fighter that the agent will be going up against plus some other context information if necessary. Then move these ROMS into the game files inside of retro like when preparing the game files after the initial cloning of the repo. Once inside that repo each state should be zipped independently of one another. Once this happens the extension will now be .zip, remove this from the extension so that the extension still remains .state. The states are now ready to be loaded by the agent. Everytime you load up the emulator decrement all the way back to zero again. 

---
## References:
https://github.com/openai/retro/issues/33 (outdated but helpful)
