# Getting Started

This readme will take you through how to get this repo up and running with the example agents, make your own test agents, and also how to create your own save states to test your agents on. 

---
## Installing Dependancies

This code only works with Python 3.6 or higher. Before trying to install dependancies it is recommend to open the terminal and run:  
`sudo apt-get update`  
`sudo apt-get upgrade`
``
To download the necessary dependancies after cloning the repo call:
`pip3 install -r requirements.txt`
This should be called in the top level directory of the repo. This will install the following libraries you will need to create game environments that serve as a wrapper abstracting the interface between your agent and the underlying emulator: 

**-gym**  
**-gym-retro**   
**-retro**  

And it will install the following libraries you will need to create and train your agent:

**-tensorflow**  

These libraries can sometimes have serious issues installing themselves or their dependancies on a windows machine. It is recommended to work on Linux. The server we will be training on runs Linux and all libraries plus code have been confirmed to work on Ubuntu's latest stable distribution.

---
## Preparing the Game Files 

After the dependancies have been installed the necessary game files, all zipped inside of **StreetFighterIISpecialChampionEdition-Genesis.zip**, can be setup. The game files need to be extracted into the actual game data files inside the installation of the retro library on your local machine. This location can be found by running the following lines in the command line:  

`python3`  
`import retro`  
`print(retro.__file__)`    

That should return the path to where the retro __init__.py script is stored. One level up from that should be the data folder. Inside there should be the stable folder. Extract the **StreetFighterIISpecialChampionEdition-Genesis.zip** here into a folder of the same name. There is already an existing folder here but we are going to overwrite that so tell your computer to replace the existing files when the prompt comes up. Inside the folder should be the following files:

-rom.md    
-rom.sha    
-scenario.json  
-data.json  
-metadata.json  
-Several .state files with each having the name of a specifc fighter from the game  

With that the game files should be correctly set up. 

---
## The first test run

To double check that the game files were properly set up the example agent can be run. cd into the src directory. Then either run the following command on your terminal:

`python3 Agent.py`

Or you can open Agent.py and excute it from your prefered IDE of choice.

---
## How to make an agent

### Agent class

### Reward Function

### Done

---
## Generating New Save States

### Installing the Emulator

### Preparing the Cores

### Saving states

---
## References:
https://github.com/openai/retro/issues/33
