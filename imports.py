import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from shutil import copyfile # keep track of generations
from stable_baselines3.common.logger import TensorBoardOutputFormat
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env