"""Implementation of the Soft Actor-Q algorithm."""

import torch
from torch.nn.functional import mse_loss
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.models.actor_critic.constraint_actor_q_critic_max_q import MaxQActorQCritic


@registry.register
class SAQ(SAC):
    """The Soft Actor-Q (SAQ) algorithm.
    Choose the discrete action by the max Q value, similar to TD3AQ.
    """

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
            next_a_cont = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_a_cont_logp = self._actor_critic.actor.log_prob(next_a_cont)
            next_q1_values_r, next_q2_values_r = self._actor_critic.target_reward_critic(next_obs, next_a_cont)

            next_q1_value_r = torch.max(next_q1_values_r, 1, keepdim=True)[0].squeeze()
            next_q2_value_r = torch.max(next_q2_values_r, 1, keepdim=True)[0].squeeze()
            next_q_value_r = torch.min(next_q1_value_r, next_q2_value_r) - next_a_cont_logp * self._alpha

            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        a_cont = action[:, 1:]  # the first dimension is batch size
        a_disc = action[:, :1]
        q1_values_r, q2_values_r = self._actor_critic.reward_critic(obs, a_cont)
        q1_value_r = q1_values_r.gather(1, a_disc.long()).squeeze()
        q2_value_r = q2_values_r.gather(1, a_disc.long()).squeeze()
        loss = mse_loss(q1_value_r, target_q_value_r) + mse_loss(q2_value_r, target_q_value_r)

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
            },
        )

    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:

        a_cont = self._actor_critic.actor.predict(obs, deterministic=False)
        a_cont_logp = self._actor_critic.actor.log_prob(a_cont)
        q1_values_r, q2_values_r = self._actor_critic.reward_critic(obs, a_cont)
        q1_value_r = torch.max(q1_values_r, 1, keepdim=True)[0].squeeze()
        q2_value_r = torch.max(q2_values_r, 1, keepdim=True)[0].squeeze()
        return (self._alpha * a_cont_logp - torch.min(q1_value_r, q2_value_r)).mean()
