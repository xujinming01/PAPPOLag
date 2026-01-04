from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import safety_gymnasium
import torch
from gymnasium.wrappers import ClipAction
from gymnasium.wrappers import RescaleObservation

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU

import omnisafe.envs.coopt_envs


@env_register
class SafeCOOPTEnv(CMDP):
    """Safety Co-opt of ACC and EMS Gymnasium Environment.

    Args:
        env_id (str): Environment id.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Keyword Args:
        render_mode (str, optional): The render mode ranges from 'human' to 'rgb_array' and 'rgb_array_list'.
            Defaults to 'rgb_array'.
        camera_name (str, optional): The camera name.
        camera_id (int, optional): The camera id.
        width (int, optional): The width of the rendered image. Defaults to 256.
        height (int, optional): The height of the rendered image. Defaults to 256.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False

    _support_envs: ClassVar[list[str]] = [
        # v0: continuous-discrete, CMDP, accel control
        'COCHTCLT-v0',
        'COWHVC-v0',
        'COWHVC3000-v0',
        'COHDUDDS-v0',
        'COJE05-v0',
        'COHHDDT-v0',

        # v999: continuous-discrete, CMDP, accel control, for mpc
        'COCHTCLT-v999',
        'COWHVC-v999',
        'COHDUDDS-v999',
        'COJE05-v999',
        'COHHDDT-v999',

        # v1: continuous, CMDP
        'COCHTCLT-v1',
        'COWHVC-v1',
        'COWHVC3000-v1',
        'ACCWHVC-v1',
        'ACCCHTCLT-v1',
        'ACCWHVC3000-v1',
        'CutIn-v1',
        'SeqEMSCHTCLT-v1',
        'SeqACCCHTCLT-v1',

        # v2: continuous-discrete
        'COCHTCLT-v2',
        'COWHVC-v2',
        'COWHVC3000-v2',

        # v3: continuous
        'COCHTCLT-v3',
        'COWHVC-v3',
        'COWHVC3000-v3',

        # v4: continuous, CMDP, torque control
        'COCHTCLT-v4',
        'COWHVC-v4',
        'COWHVC3000-v4',

        # v5: continuous, torque control
        'COCHTCLT-v5',
        'COWHVC-v5',
        'COWHVC3000-v5',

        # v6: continuous-discrete, CMDP, torque control
        'COCHTCLT-v6',
        'COWHVC-v6',
        'COWHVC3000-v6',
        'COHDUDDS-v6',

        # v7: continuous-discrete, torque control
        'COCHTCLT-v7',
        'COWHVC-v7',
        'COWHVC3000-v7',
        'COHDUDDS-v7',
        'COJE05-v7',
        'COHHDDT-v7',

        # v70: continuous-discrete, torque control, for mpc
        'COCHTCLT-v70',
        'COWHVC-v70',
        'COWHVC3000-v70',
        'COHDUDDS-v70',
        'COJE05-v70',
        'COHHDDT-v70',
    ]

    def __init__(
            self,
            env_id: str,
            num_envs: int = 1,
            device: torch.device = DEVICE_CPU,
            **kwargs: Any,
    ) -> None:
        """Initialize an instance of :class:`SafetyGymnasiumEnv`."""
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        if num_envs == 1:
            # set healthy_reward=0.0 for removing the safety constraint in reward
            self._env = safety_gymnasium.make(id=env_id, autoreset=False, **kwargs)
            # self._env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(self._env)
            # if self._env.action_space.__class__.__name__ == 'Box':
            #     self._env = ClipAction(self._env)  # NOTE: ClipAction is indispensable for PPO
            # obs_shape = self._env.observation_space.shape
            # self._env = RescaleObservation(
            #     self._env,
            #     min_obs=np.full(obs_shape, -1.0, dtype=np.float32),
            #     max_obs=np.full(obs_shape, 1.0, dtype=np.float32),
            # )
            # self._env = safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(self._env)
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space

            self.need_auto_reset_wrapper = True
        else:
            raise NotImplementedError('Only support num_envs=1 now.')
        self._metadata = self._env.metadata

        # self.env_spec_log = self._env.env.env.addi_penalty

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Step the environment.

        .. note::
            OmniSafe uses auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode. And the
            true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key
            of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.


        Returns:
            observation: Agent's observation of the current environment.
            info: Some information logged by the environment.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32,
                               device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def sample_action(self) -> torch.Tensor:
        """Sample a random action.

        Returns:
            A random action.
        """
        return torch.as_tensor(
            self._env.action_space.sample(),
            dtype=torch.float32,
            device=self._device,
        )

    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            The render frames: we recommend to use `np.ndarray`
                which could construct video by moviepy.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    # def reset_zero(self, soc_init) -> torch.Tensor:
    #     """Reset the environment to zero state.
    #
    #     Args:
    #         soc_init (float): The initial soc.
    #     Returns:
    #         The observation of the environment.
    #     """
    #     obs = self._env.reset_zero(soc_init)
    #     return torch.as_tensor(obs, dtype=torch.float32, device=self._device)
    #
    # def evaluate_on(self):
    #     """Turn on the evaluation mode."""
    #     self._env.evaluate_on()

    # def spec_log(self, logger):
    #     for k, v in self._env.env.env.addi_penalty.items():
    #         logger.store({f'{k}': v})
    #         self._env.env.env.addi_penalty[k] = 0

