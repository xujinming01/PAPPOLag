"""Implementation of the SAQLag algorithm."""

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac_lag import SACLag
from omnisafe.algorithms.off_policy.saq import SAQ


@registry.register
class SAQLag(SAQ, SACLag):

    # def _update_cost_critic(
    #     self,
    #     obs: torch.Tensor,
    #     action: torch.Tensor,
    #     cost: torch.Tensor,
    #     done: torch.Tensor,
    #     next_obs: torch.Tensor,
    # ) -> None:
    #     with torch.no_grad():
    #         n_a_cont = self._actor_critic.actor.predict(next_obs, deterministic=False)
    #         n_q_values_c = self._actor_critic.target_cost_critic(next_obs, n_a_cont)[0]
    #         n_q_value_c = torch.min(n_q_values_c, 1, keepdim=True)[0].squeeze()
    #         target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * n_q_value_c
    #     a_cont = action[:, 1:]
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

    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:
        a_cont = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(a_cont)
        losses_q_r_1, losses_q_r_2 = self._actor_critic.reward_critic(obs, a_cont)
        loss_q_r_1 = torch.max(losses_q_r_1, 1, keepdim=True)[0].squeeze()
        loss_q_r_2 = torch.max(losses_q_r_2, 1, keepdim=True)[0].squeeze()
        loss_q_r = torch.min(loss_q_r_1, loss_q_r_2)
        loss_r = self._alpha * log_prob - loss_q_r
        # losses_q_c = self._actor_critic.cost_critic(obs, a_cont)[0]
        # loss_q_c = torch.min(losses_q_c, 1, keepdim=True)[0].squeeze()
        loss_q_c = self._actor_critic.cost_critic(obs, a_cont)[0]
        loss_c = self._lagrange.lagrangian_multiplier.item() * loss_q_c

        return (loss_r + loss_c).mean() / (1 + self._lagrange.lagrangian_multiplier.item())
