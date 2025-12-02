from copy import deepcopy

import torch
from torch import optim

from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig

from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic


class MaxQActorQCritic(ConstraintActorQCritic):

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorQCritic`."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)

        self.cost_critic: Critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
            separate_cost=True,  # output 1 value for cost network
        ).build_critic('q')
        self.target_cost_critic: Critic = deepcopy(self.cost_critic)
        for param in self.target_cost_critic.parameters():
            param.requires_grad = False
        self.add_module('cost_critic', self.cost_critic)
        if model_cfgs.critic.lr is not None:
            self.cost_critic_optimizer: optim.Optimizer
            self.cost_critic_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose action based on observation.
        The continuous action is chosen by the actor directly,
        and the discrete action is chosen by the max Q value.
        """
        with torch.no_grad():
            action_cont = self.actor.predict(obs, deterministic=deterministic)
            q1_values, q2_values = self.reward_critic(obs, action_cont)
            action_disc = torch.argmax(q1_values, dim=1).unsqueeze(1)

            return torch.cat((action_disc, action_cont), dim=1)
