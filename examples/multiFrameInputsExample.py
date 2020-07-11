import retro
from enum import Enum

class multiFrameInput():
  
    def __init__(self, frames):
        self.frames = frames

    def __iter__(self):
        self.frameIndex = 0
        return self

    def __next__(self):
        if self.frameIndex < len(self.frames):
            frameInput = self.frames[self.frameIndex]
            self.frameIndex += 1
            return frameInput
        else:
            raise(StopIteration)

class Moves(Enum):
    Hadoken = 0
    Tetsumaki = 1
    DragonUppercut = 2

specialMovesDict = {
    Moves.Hadoken :    [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]],
    Moves.Tetsumaki : [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    Moves.DragonUppercut : [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
}

frameIndex = 0
def getCurrentAction():
    global frameIndex
    currentAction = specialMovesDict[Moves.Tetsumaki]
    frameInput = currentAction[frameIndex]
    frameIndex = (frameIndex + 1) % len(currentAction)
    return frameInput

def main(game= 'StreetFighterIISpecialChampionEdition-Genesis',  state= "chunli"):
    env = retro.make(game= game, state= state)
    env.reset()
    while True:
        action = getCurrentAction()
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
    env.close()

if __name__ == "__main__":
    main()
