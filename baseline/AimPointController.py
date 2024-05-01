import sys
sys.path.insert(0, 'C:/Users/Kangy/deep-rl-supertux-race')
import numpy as np
import torch
from typing import Dict, Optional, Tuple

from agents.AbstractAgent import AbstractAgent
from baseline.planner import load_model
from environments.pytux import VELOCITY, IMAGE, PyTux

class AimPointController(AbstractAgent):
    def __init__(self, env: PyTux, options: Optional[Dict] = None, disable_drift: bool = False, fix_vel: Optional[float] = None):
        super().__init__(env, options)
        self.aim_planner = load_model().eval()
        self.disable_drift = disable_drift
        self.fix_vel = fix_vel

    def act(self, state: PyTux.State, noise: Optional[Tuple[float, float]] = (0, 0)) -> PyTux.Action:
        """
        Calculates and returns an action based on the current state and optional noise parameters.
        
        Parameters:
            state : PyTux.State
                The current state from the PyTux simulator.
            noise : Tuple[float, float], optional
                A tuple of noise values for the aim point and velocity, respectively.

        Returns:
            PyTux.Action
                The action to be executed based on the aim point and velocity.
        """
        super().act(state)  # Ensure this call is correct if the superclass requires specific handling

        img = self._to_torch(state[IMAGE])
        vel = np.linalg.norm(state[VELOCITY]) + noise[1] * np.random.randn()
        aim_point = self._predict_aim_point(img)
        aim_point += noise[0] * np.random.randn(*aim_point.shape)

        return self._control(aim_point, vel)

    def _to_torch(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy image array to a Torch tensor, adjusting dimensions for model input.
        """
        return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.aim_planner.device)

    def _predict_aim_point(self, img: torch.Tensor) -> np.ndarray:
        """
        Predict the aim point using the loaded CNN model from a given image tensor.
        """
        with torch.no_grad():
            return self.aim_planner(img).squeeze(0).cpu().numpy()

    def _control(self, aim_point: np.ndarray, current_vel: float) -> PyTux.Action:
        """
        Generate the driving action based on the predicted aim point and current velocity.
        
        Parameters:
            aim_point : np.ndarray
                The predicted aim point from the CNN.
            current_vel : float
                The current velocity of the vehicle.

        Returns:
            PyTux.Action
                The computed action for the vehicle.
        """
        action = PyTux.Action()
        abs_x = np.abs(aim_point[0])
        MIN_SPEED = 8
        CRITICAL_SPEED = 35
        TURN_THRESHOLD = 0.4
        DRIFT_THRESHOLD = 0.2
        BRAKE_THRESHOLD = 0.4
        NITRO_THRESHOLD = 0.5

        action.acceleration = 1 if (current_vel < CRITICAL_SPEED and abs_x < TURN_THRESHOLD) or current_vel < MIN_SPEED else 0
        action.steer = np.tanh(current_vel * aim_point[0])
        action.drift = abs_x > DRIFT_THRESHOLD or (abs_x > 0.15 and current_vel > 15)
        action.brake = current_vel > 20 and abs_x > BRAKE_THRESHOLD
        action.nitro = abs_x < NITRO_THRESHOLD

        if self.disable_drift:
            action.drift = False
        if self.fix_vel is not None:
            action.acceleration = self.fix_vel

        return action