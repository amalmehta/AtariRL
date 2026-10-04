"""The Atari environment: the emulator plus the paper's preprocessing function phi.

We talk to the Arcade Learning Environment (ALE) directly rather than through
Gymnasium so every step the paper describes is visible here:

* grey-scale the 210x160 screen,
* down-sample it to 110x84,
* crop an 84x84 square that covers the playing area,
* repeat each chosen action for k frames (frame skipping) and sum the reward.
"""
import numpy as np
from PIL import Image
from ale_py import ALEInterface, LoggerMode, roms

# Rows of the 110x84 image kept by the crop. The playing area of most games
# sits between screen rows ~34 and ~194, i.e. rows ~18..102 after resizing.
CROP_TOP = 18
CROP_SIZE = 84

ALEInterface.setLoggerMode(LoggerMode.Error)


def preprocess(gray_screen: np.ndarray) -> np.ndarray:
    """phi for one frame: (210, 160) uint8 grey -> (84, 84) uint8."""
    img = Image.fromarray(gray_screen).resize((84, 110), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)[CROP_TOP:CROP_TOP + CROP_SIZE]


class AtariEnv:
    def __init__(self, game: str, frame_skip: int = 4, seed: int = 0):
        self.game = game
        self.frame_skip = frame_skip
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", seed)
        # The 2013 setup is deterministic: no sticky actions.
        self.ale.setFloat("repeat_action_probability", 0.0)
        self.ale.setInt("frame_skip", 1)
        self.ale.loadROM(roms.get_rom_path(game))
        # Only the actions that do something in this game (the paper's "legal actions").
        self.actions = list(self.ale.getMinimalActionSet())
        self.n_actions = len(self.actions)
        self._gray = np.empty((210, 160), dtype=np.uint8)

    def _frame(self) -> np.ndarray:
        self.ale.getScreenGrayscale(self._gray)
        return preprocess(self._gray)

    def reset(self) -> np.ndarray:
        self.ale.reset_game()
        return self._frame()

    def step(self, action_index: int):
        """Repeat the action for k frames. Returns (frame, raw_reward, done)."""
        action = self.actions[action_index]
        reward = 0.0
        for _ in range(self.frame_skip):
            reward += self.ale.act(action)
            if self.ale.game_over():
                break
        return self._frame(), reward, self.ale.game_over()

    def rgb_screen(self) -> np.ndarray:
        """The raw colour screen, for recording videos."""
        return self.ale.getScreenRGB()
