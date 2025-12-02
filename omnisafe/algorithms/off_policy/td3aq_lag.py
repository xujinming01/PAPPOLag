"""Implementation of the TD3AQLag algorithm."""

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.td3aq import TD3AQ
from omnisafe.algorithms.off_policy.td3_lag import TD3Lag


@registry.register
class TD3AQLag(TD3AQ, TD3Lag):
    """The Lagrangian-based Twin Delayed DDPG Actor-Q (TD3AQLag) algorithm."""

    # def _update_cost_critic(
    #     self,
    #     obs: torch.Tensor,
    #     action: torch.Tensor,
    #     cost: torch.Tensor,
    #     done: torch.Tensor,
    #     next_obs: torch.Tensor,
    # ) -> None:
    #     """Update cost critic.
    #
    #     - Get the TD loss of cost critic.
    #     - Update critic network by loss.
    #     - Log useful information.
    #
    #     Args:
    #         obs (torch.Tensor): The ``observation`` sampled from buffer.
    #         action (torch.Tensor): The ``action`` sampled from buffer.
    #         cost (torch.Tensor): The ``cost`` sampled from buffer.
    #         done (torch.Tensor): The ``terminated`` sampled from buffer.
    #         next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
    #     """
    #     with torch.no_grad():
    #         next_a_cont = self._actor_critic.actor.predict(next_obs, deterministic=True)
    #         next_q_values_c = self._actor_critic.target_cost_critic(next_obs, next_a_cont)[0]
    #         next_q_value_c = torch.min(next_q_values_c, 1, keepdim=True)[0].squeeze()
    #         target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_c
    #     a_cont = action[:, 1:]  # the first dimension is batch size
    #     a_disc = action[:, :1]
    #     q_values_c = self._actor_critic.cost_critic(obs, a_cont)[0]
    #     q_value_c = q_values_c.gather(1, a_disc.long()).squeeze()
    #     loss = nn.functional.mse_loss(q_value_c, target_q_value_c)
    #
    #     if self._cfgs.algo_cfgs.use_critic_norm:
    #         for param in self._actor_critic.cost_critic.parameters():
    #             loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff
    #
    #     self._actor_critic.cost_critic_optimizer.zero_grad()
    #     loss.backward()
    #
    #     if self._cfgs.algo_cfgs.max_grad_norm:
    #         clip_grad_norm_(
    #             self._actor_critic.cost_critic.parameters(),
    #             self._cfgs.algo_cfgs.max_grad_norm,
    #         )
    #     self._actor_critic.cost_critic_optimizer.step()
    #
    #     self._logger.store(
    #         {
    #             'Loss/Loss_cost_critic': loss.mean().item(),
    #             'Value/cost_critic': q_value_c.mean().item(),
    #         },
    #     )

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        a_cont = action[:, 1:]
        super()._update_cost_critic(obs, a_cont, cost, done, next_obs)


    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        action_conti = self._actor_critic.actor.predict(obs, deterministic=True)
        q1_values_r, q2_values_r = self._actor_critic.reward_critic(obs, action_conti)
        loss_r = -torch.max(q1_values_r, 1, keepdim=True)[0].squeeze()  # maximize the reward
        # q_values_c = self._actor_critic.cost_critic(obs, action_conti)[0]
        # loss_q_c = torch.min(q_values_c, 1, keepdim=True)[0].squeeze()  # minimize the cost
        loss_q_c = self._actor_critic.cost_critic(obs, action_conti)[0]
        loss_c = self._lagrange.lagrangian_multiplier.item() * loss_q_c

        return (loss_r + loss_c).mean() / (1 + self._lagrange.lagrangian_multiplier.item())
