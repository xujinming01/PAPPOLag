"""Implementation of the TD3AQ algorithm."""

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.models.actor_critic.constraint_actor_q_critic_max_q import MaxQActorQCritic


@registry.register
class TD3AQ(DDPG):
    """The Twin Delayed DDPG Actor-Q (TD3AQ) algorithm."""

    def _init_model(self) -> None:
        """Initialize the model.

        The ``num_critics`` in ``critic`` configuration must be 2.
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 2
        self._actor_critic = MaxQActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_a_cont = self._actor_critic.target_actor.predict(next_obs, deterministic=True)
            policy_noise = self._cfgs.algo_cfgs.policy_noise
            policy_noise_clip = self._cfgs.algo_cfgs.policy_noise_clip
            noise = (torch.randn_like(next_a_cont) * policy_noise).clamp(-policy_noise_clip, policy_noise_clip)
            next_a_cont = (next_a_cont + noise).clamp(-1.0, 1.0)
            next_q1_values_r, next_q2_values_r = self._actor_critic.target_reward_critic(next_obs, next_a_cont)

            next_q1_value_r = torch.max(next_q1_values_r, 1, keepdim=True)[0].squeeze()
            next_q2_value_r = torch.max(next_q2_values_r, 1, keepdim=True)[0].squeeze()
            next_q_value_r = torch.min(next_q1_value_r, next_q2_value_r)
            # next_q_value_r = torch.min(next_q1_values_r.max(dim=1)[0], next_q2_values_r.max(dim=1)[0])  # another way

            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        a_cont = action[:, 1:]  # the first dimension is batch size
        a_disc = action[:, :1]
        q1_values_r, q2_values_r = self._actor_critic.reward_critic(obs, a_cont)
        q1_value_r = q1_values_r.gather(1, a_disc.long()).squeeze()
        q2_value_r = q2_values_r.gather(1, a_disc.long()).squeeze()
        loss = F.mse_loss(q1_value_r, target_q_value_r) + F.mse_loss(q2_value_r, target_q_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q1_value_r.mean().item(),
            }
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Choose the maximum Q value of Q1 as the loss of the policy."""
        action_conti = self._actor_critic.actor.predict(obs, deterministic=True)
        q1_values_r, q2_values_r = self._actor_critic.reward_critic(obs, action_conti)
        q_value_max = torch.max(q1_values_r, 1, keepdim=True)[0].squeeze()
        return -q_value_max.mean()
