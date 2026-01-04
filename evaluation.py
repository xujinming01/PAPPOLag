from __future__ import annotations

import os
import itertools
import time

import scipy.io as sio
import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm

import omnisafe
from omnisafe.evaluator import Evaluator

from utils import CYCLES, least_squares, DT


def evaluation_all(log_dir: str, save_mat: bool = False):
    """Evaluate the saved models to determine the best model, note the model
    will be evaluated on the TRAINED env.

    Args:
        log_dir: dir like '.../seed-000-2023-03-07-20-25-48'.
        save_mat: whether to save the evaluation results to mat files.
    """
    model_results = {
        'model_name': [],
        'ep_rews': [],
        'ep_costs': [],
    }
    evaluator = Evaluator()
    scan_dir = os.scandir(os.path.join(log_dir, 'torch_save'))
    for item in scan_dir:
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            evaluator.load_saved(
                save_dir=log_dir,
                model_name=item.name,
            )
            ep_rew, ep_cost, ep_info = evaluator.evaluate(num_episodes=1)
            if save_mat:
                mat_dir = os.path.join(log_dir, 'eval_mat')  # save to mat_dir
                os.makedirs(mat_dir, exist_ok=True)
                sio.savemat(os.path.join(mat_dir, item.name + '.mat'), ep_info)
            model_results['model_name'].append(item.name)
            model_results['ep_rews'].append(np.mean(ep_rew))
            model_results['ep_costs'].append(np.mean(ep_cost))
            print(f'Evaluate {item.name} done! ')
    scan_dir.close()

    # choose the lowest cost model with the highest reward
    ep_rews = model_results['ep_rews']
    ep_costs = model_results['ep_costs']
    model_names = model_results['model_name']

    # sort the rewards and costs
    sorted_idx = np.argsort(ep_costs)
    sorted_rews = np.array(ep_rews)[sorted_idx]
    sorted_costs = np.array(ep_costs)[sorted_idx]
    sorted_model_names = np.array(model_names)[sorted_idx]

    # get the indices of the lowest cost models
    min_cost_idx = np.where(sorted_costs == np.min(sorted_costs))[0]

    # get the indices of the highest reward models
    chosen_idx = min_cost_idx[np.argmax(sorted_rews[min_cost_idx])]

    best_rew = sorted_rews[chosen_idx]
    best_cost = sorted_costs[chosen_idx]
    best_model = sorted_model_names[chosen_idx]

    # print the model name
    print(f'\nBest model: {best_model}')
    print(f'Best reward: {best_rew:.2f}')
    print(f'Best cost: {best_cost:.2f}')

    # print the results of the model
    res_best = sio.loadmat(os.path.join(log_dir, 'eval_mat', best_model + '.mat'))
    print(f'\tep_length: {len(res_best["mf"][0])}')
    print(f'\t|jerk|.mean [m/s³]: {np.abs(res_best["jerk"]).mean():.3f}')
    print(f'\t|ed|.mean [m]: {np.abs(res_best["ed"]).mean():.3f}')
    print(f'\tfuel consumption [kg]: {res_best["mf"].sum():.3f}')
    print(f'\tSOC final: {res_best["soc"][0][-1]:.4f}')
    # print(f'\tcurrency [CNY]: {res_best["currency"].sum():.2f}')
    # print(f'\tp_vh: {res_best["p_vh"].sum()}')
    # print(f'\tp_collision: {res_best["p_collision"].sum()}')
    print(f'\tp_ed: {res_best["p_ed"].sum()}')
    print(f'\tp_w: {res_best["p_w"].sum()}')
    print(f'\tp_Tm: {res_best["p_Tm"].sum()}')
    print(f'\tp_soc: {res_best["p_soc"].sum()}')
    print(f'\tr_ed: {res_best["r_ed"].sum():.2f}')
    print(f'\tr_jerk: {res_best["r_jerk"].sum():.2f}')
    print(f'\tr_mf: {res_best["r_mf"].sum():.4f}')
    print(f'\tr_soc: {res_best["r_soc"].sum():.4f}')
    print(f'\tr_Ah: {res_best["r_Ah"].sum()}')

    return best_rew, best_cost, best_model


def evaluation_one(log_dir: str, model_name: str, save_mat: bool = False, eval_env: str = None):
    """Evaluate the designated model."""
    # assert DT == 0.1, "DT should be 0.1 for evaluation."
    evaluator = omnisafe.Evaluator()
    evaluator.load_saved(
        save_dir=log_dir,
        model_name=model_name,
    )
    start_time = time.time()
    ep_rew, ep_cost, ep_info = evaluator.evaluate(num_episodes=1)
    print(f'Average time per step: {(time.time() - start_time)/len(ep_info["mf"])*1000:.2f} ms')
    if save_mat:
        mat_dir = os.path.join(log_dir, 'eval_mat')  # save to mat_dir
        os.makedirs(mat_dir, exist_ok=True)
        sio.savemat(os.path.join(mat_dir, model_name + '.mat'), ep_info)
    r_ed, r_ed_ref = np.sum(ep_info['r_ed']), CYCLES[eval_env]['r_ed']
    r_jerk, r_jerk_ref = np.sum(ep_info['r_jerk']), CYCLES[eval_env]['r_j']
    r_Ah, r_Ah_ref = np.sum(ep_info['r_Ah']), CYCLES[eval_env]['r_Ah']

    print(f'Fuel consumption [kg]: {np.sum(ep_info["mf"]):.3f}')
    print(f'Final SOC: {ep_info["soc"][-1]:.4f}')
    print(f'Mean of |ed| [m]: {np.abs(ep_info["ed"]).mean():.3f}')
    print(f'Mean of |jerk| [m/s³]: {np.abs(ep_info["jerk"]).mean():.3f}')
    # print(f'{r_ed:.2f} ({r_ed/r_ed_ref-1:.2%}) | '
    #       f'{r_jerk:.2f} ({r_jerk/r_jerk_ref-1:.2%}) | '
    #       f'{r_Ah:.2f} ({r_Ah/r_Ah_ref-1:.2%}) | '
    #       f'{np.sum(ep_info["r_mf"]):.2f} | '
    #       f'{np.sum(ep_info["r_soc"]):.2f}')
    print('r_ed | r_j | r_mf | r_soc | r_Ah | c_ed | c_Tm | c_soc | shift:')
    print(
        f'{r_ed:.2f} | '
        f'{r_jerk:.2f} | '
        f'{np.sum(ep_info["r_mf"]):.2f} | '
        f'{np.sum(ep_info["r_soc"]):.2f} | '
        f'{r_Ah:.2f} | '
        f'{np.sum(ep_info["p_ed"])} | '
        f'{np.sum(ep_info["p_Tm"]):.2f} | '
        f'{np.sum(ep_info["p_soc"]):.2f} | '
        f'{ep_info["shift_count"][-1]}'
    )


    return ep_rew, ep_cost, ep_info


class NewEnvEval(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_saved(
            self,
            save_dir: str,
            model_name: str,
            env_id: str,
    ) -> None:
        """Load the env that can be not trained.

        Args:
            save_dir: dir like '.../seed-000-2023-03-07-20-25-48'.
            model_name: name like 'epoch-100.pt'.
            env_id: the env id like 'IDDWHVC-v3'.
        """
        # load the config
        self._save_dir = save_dir
        self._model_name = model_name

        self._Evaluator__load_cfgs(save_dir)

        env_kwargs = {
            'env_id': env_id,
            'num_envs': 1,
        }

        self._Evaluator__load_model_and_env(save_dir, model_name, env_kwargs)

    def traced_predict_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict the actor action given the observation"""
        with torch.no_grad():
            for param in self._actor.net.parameters():
                param.requires_grad_(False)
            if self._actor is not None:
                act_cont = self._actor.predict(obs, deterministic=True)
            else:
                raise ValueError(
                    'The policy must be provided or created before evaluating the agent.',
                )
        return act_cont

    def traced_predict_critic(self, obs_act: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._actor is not None:
                for critic in self._critic.net_lst:
                    for param in critic.parameters():
                        param.requires_grad_(False)
                # act_cont = self._actor.predict(obs, deterministic=True)
                # obs_act = torch.cat([obs, act_cont], dim=-1)
                # q1, q2 = self._critic(obs, act_cont)
                q1 = self._critic.traced_predict(obs_act)
                # act_disc = torch.argmax(q1).unsqueeze(0)
            else:
                raise ValueError(
                    'The policy must be provided or created before evaluating the agent.',
                )
        return q1


def equivalent_fuel_eval(save_dir: str, model_name: str, eval_envs: list[str]):
    """
    Evaluate the equivalent fuel consumption on the given cycles.

    Args:
        save_dir: dir like '.../seed-000-2023-03-07-20-25-48'.
        model_name: model name like 'epoch-100.pt'.
        eval_envs: the list of the env-ids.
    """
    assert DT == 0.1, "DT must be 0.1 for equivalent fuel evaluation."
    Qmax = 26  # battery capacity, Ah
    Ic_max = 10
    I_max = Ic_max * Qmax  # max current, A
    Ah_max = I_max * DT / 3600  # max Ah throughout

    evaluator = NewEnvEval()
    eval_costs = []
    eval_gaps = []
    for env_id in eval_envs:
        evaluator.load_saved(
            save_dir=save_dir,
            model_name=model_name,
            env_id=env_id,
        )

        EP_REW, EP_COST, EP_INFO = evaluator.evaluate(num_episodes=1)
        # save the evaluation results
        mat_dir = os.path.join(save_dir, 'generalization_mat')
        os.makedirs(mat_dir, exist_ok=True)
        sio.savemat(os.path.join(mat_dir, env_id + '.mat'), EP_INFO)

        # calculate the equivalent fuel consumption
        SOC_0 = 0.6
        SOC_H = 0.65
        SOC_L = 0.55
        DELTA_SOC = EP_INFO['soc'][-1] - SOC_0
        FUEL_UN_EQ = np.sum(EP_INFO['mf'])
        delta_soc_list = np.array([DELTA_SOC])
        fuel_list = FUEL_UN_EQ

        # collect 2N groups of random initial SOC
        N = 6
        assert N >= 6
        # SOC_0 to soc_max
        soc_0_list_h = npr.uniform(SOC_0, SOC_H, N)
        # soc_min to SOC_0
        soc_0_list_l = npr.uniform(SOC_L, SOC_0, N)

        # soc_0_list = np.round(np.concatenate((soc_0_list_h, soc_0_list_l)), 6)
        soc_0_list = [0.606406, 0.611804, 0.619829, 0.596777, 0.565664, 0.576227]

        selected = isinstance(soc_0_list, np.ndarray)

        for soc_init in tqdm(soc_0_list):
            evaluator.load_saved(
                save_dir=save_dir,
                model_name=model_name,
                env_id=env_id,
            )
            ep_rew, ep_cost, ep_info = evaluator.evaluate(
                num_episodes=1,
                soc_init=soc_init,
            )
            delta_soc_list = np.append(delta_soc_list, ep_info['soc'][-1] - soc_init)
            fuel_list = np.append(fuel_list, np.sum(ep_info['mf']))
        # delta_soc_list = [-0.093858, -0.093212, -0.085167, -0.093876, -0.086247, -0.087852, -0.077765, -0.080583, -0.083241, -0.082349, -0.077019, -0.079131, -0.083886]
        # fuel_list = [1.5437, 1.545516, 1.559835, 1.543668, 1.556435, 1.553, 1.577782, 1.568991, 1.563894, 1.565419, 1.578458, 1.572932, 1.562359]

        print(f'\nEvaluate {env_id} done! ')
        print(f'Random initial SOC: {soc_0_list}')
        print(f'delta_soc_list: {[round(s, 6) for s in delta_soc_list]}')
        print(f'fuel_list: {[round(f, 6) for f in fuel_list]}')

        # fuel = a + b * delta_soc
        a, b, R2, xy_lim = least_squares(x=delta_soc_list, y=fuel_list, xy_lim='scale')

        env = env_id.split('-')[0]
        delta_soc_ref = CYCLES[env]['soc_final_ref'] - SOC_0
        FUEL_EQ = a + b * delta_soc_ref

        print(f'R2 = {R2:.4f}')
        print(f'y (fuel_eq) = {a:.4f} + {b:.4f} * x (SOC_T - SOC_0)')
        print(f'\nUn normalized fuel consumption: {FUEL_UN_EQ:.4f}')
        print(f'SOC final: {EP_INFO["soc"][-1]:.4f}')
        print(f'Equivalent fuel consumption: {FUEL_EQ:.4f}')
        print(f'Gap: {FUEL_EQ / CYCLES[env]["ref_optimal"] - 1:.2%}')
        print(f'Cost: {np.mean(EP_COST):.2f}')
        # print(f'Mean of |ed| [m]: {np.abs(EP_INFO["ed"]).mean():.3f}')
        # print(f'Mean of |jerk| [m/s³]: {np.abs(EP_INFO["jerk"]).mean():.3f}')

        eval_costs.append(np.mean(EP_COST))
        eval_gaps.append(FUEL_EQ / CYCLES[env]["ref_optimal"] - 1)

        # select half the data
        if selected:
            # remove the original data
            delta_soc_list = np.delete(delta_soc_list, 0)
            fuel_list = np.delete(fuel_list, 0)
            assert len(delta_soc_list) == len(fuel_list) == len(soc_0_list)
            combs = itertools.combinations(range(N), 3)
            min_fuel_eq = float('inf')
            best_comb = None
            delta_soc_h = delta_soc_list[:N]
            fuel_h = fuel_list[:N]
            delta_soc_l = delta_soc_list[N:]
            fuel_l = fuel_list[N:]
            for comb in combs:
                idx = list(comb)
                x_comb = np.concatenate((delta_soc_h[idx], delta_soc_l[idx], np.array([DELTA_SOC])))
                y_comb = np.concatenate((fuel_h[idx], fuel_l[idx], np.array([FUEL_UN_EQ])))
                # use numpy to fit the linear regression
                A = np.vstack([x_comb, np.ones(len(x_comb))]).T
                b_comb, a_comb = np.linalg.lstsq(A, y_comb, rcond=None)[0]
                y_pred = b_comb * x_comb + a_comb
                ss_res = np.sum((y_comb - y_pred) ** 2)
                ss_tot = np.sum((y_comb - np.mean(y_comb)) ** 2)
                r_squared = 1 - ss_res / ss_tot
                fuel_eq = a_comb + b_comb * delta_soc_ref
                if fuel_eq < min_fuel_eq and r_squared > 0.8:
                    min_fuel_eq = fuel_eq
                    best_comb = comb
            idx_best = list(best_comb)
            x_best = np.concatenate((delta_soc_h[idx_best], delta_soc_l[idx_best], np.array([DELTA_SOC])))
            y_best = np.concatenate((fuel_h[idx_best], fuel_l[idx_best], np.array([FUEL_UN_EQ])))
            a_best, b_best, R2, _ = least_squares(x=x_best, y=y_best, xy_lim=xy_lim)
            fuel_eq_best = a_best + b_best * delta_soc_ref
            print(f'\nR2 = {R2:.4f}')
            print(f'y (fuel_eq) = {a_best:.4f} + {b_best:.4f} * x (SOC_T - SOC_0)')
            best_soc_0 = np.concatenate((soc_0_list_h[idx_best], soc_0_list_l[idx_best], np.array([SOC_0])))
            print(f'best init SOC: {np.round(best_soc_0, 6).tolist()}')
            print(f'best delta_soc_list: {[round(s, 6) for s in x_best]}')
            print(f'best fuel_list: {[round(f, 6) for f in y_best]}')
            print(f'SOC final: {EP_INFO["soc"][-1]:.4f}')
            print(f'Un normalized fuel consumption: {FUEL_UN_EQ:.4f}')
            print(f'best fuel_eq: {fuel_eq_best:.4f}')
            print(f'best gap: {fuel_eq_best / CYCLES[env]["ref_optimal"] - 1:.2%}')
            least_squares(x=x_best, y=y_best)

        print(f'p_ed: {np.sum(EP_INFO["p_ed"]):.2f}')
        # print(f'p_w: {np.sum(EP_INFO["p_w"]):.2f}')
        print(f'p_Tm: {np.sum(EP_INFO["p_Tm"]):.2f}')
        print(f'p_soc: {np.sum(EP_INFO["p_soc"]):.2f}')
        print(f'r_ed: {np.sum(EP_INFO["r_ed"]):.2f}')
        print(f'r_jerk: {np.sum(EP_INFO["r_jerk"]):.2f}')
        r_Ah = np.sum(EP_INFO["r_Ah"])
        Ah_eff = -r_Ah * Ah_max
        print(f'r_Ah: {r_Ah:.2f}')
        print(f'Ah_eff: {Ah_eff:.2f} Ah')
        print(f'r_mf: {np.sum(EP_INFO["r_mf"]):.2f}')
        print(f'r_soc: {np.sum(EP_INFO["r_soc"]):.2f}')

    print(f'\nEvaluate {len(eval_envs)} envs done!')
    for env_id, eval_cost, eval_gap in zip(eval_envs, eval_costs, eval_gaps):
        print(f'{env_id}: mean cost: {eval_cost:.2f}, gap: {eval_gap:.2%}')


def cut_in_eval(save_dir, model_name, eval_env, num_episodes: int = 1):
    """
    Evaluate the cut-in scenario.

    Args:
        save_dir: dir like '.../seed-000-2023-03-07-20-25-48'.
        model_name: model name like 'epoch-100.pt'.
        eval_env: the env-id like 'CutIn-v1'.
        num_episodes: int, number of episodes to evaluate.
    """
    evaluator = NewEnvEval()
    evaluator.load_saved(
        save_dir=save_dir,
        model_name=model_name,
        env_id=eval_env,
    )

    res = {
        'cost': [],
        'p_ed': [],
        'p_Tm': [],
        'p_soc': [],
        'r_ed': [],
        'r_jerk': [],
        'jerk': [],
        'n_success': 0,
    }
    for i in range(num_episodes):
        ep_rew, ep_cost, ep_info = evaluator.evaluate(num_episodes=1)
        for key in res.keys():
            if key in ep_info:
                res[key].append(ep_info[key])
        res['n_success'] += 1 if np.mean(ep_cost) == 0 else 0

        if num_episodes <= 1:  # only save one episode
            # save the evaluation results
            mat_dir = os.path.join(save_dir, 'cut_in_mat')
            os.makedirs(mat_dir, exist_ok=True)
            sio.savemat(os.path.join(mat_dir, eval_env + '.mat'), ep_info)

            from plot_coopt import plot_acc, plot_ems
            from utils import DT
            len_xt = len(ep_info['v']) - 1
            xt = np.arange(0, len_xt * DT, DT)
            mat_preceeding = {
                'vp': [ep_info['vp']],
                'ap': [np.zeros(len_xt + 1)],
            }
            mat_rl = sio.loadmat(os.path.join(mat_dir, eval_env + '.mat'))

            plot_acc(mat_preceeding, mat_rl, xt, model_name, save_to=mat_dir + '/fig-acc.png')
            # res_ems = {
            #     'Te': [mat_rl['Te'][0]],
            #     'Tm': [mat_rl['Tm'][0]],
            #     'SOC': [mat_rl['soc'][0]],
            #     'Gear': [mat_rl['gear'][0] + 1],
            # }
            # plot_ems(res_ems, xt, model_name, save_to=mat_dir + '/fig-ems.png')

            print(f'\nEvaluate {eval_env} done! ')
            print(f'Fuel consumption [kg]: {np.sum(ep_info["mf"]):.3f}')
            print(f'Final SOC: {ep_info["soc"][-1]:.4f}')
            print(f'Mean of |ed| [m]: {np.abs(ep_info["ed"]).mean():.3f}')
            print(f'Mean of |jerk| [m/s³]: {np.abs(ep_info["jerk"]).mean():.3f}')
            print(f'p_ed: {np.sum(ep_info["p_ed"])}')
            print(f'p_w: {np.sum(ep_info["p_w"])}')
            print(f'p_Tm: {np.sum(ep_info["p_Tm"])}')
            print(f'p_soc: {np.sum(ep_info["p_soc"])}')
            print(f'r_ed: {np.sum(ep_info["r_ed"]):.4f}')
            print(f'r_jerk: {np.sum(ep_info["r_jerk"]):.4f}')
    success_rate = res['n_success'] / num_episodes
    mean_cost = np.sum(res['cost']) / num_episodes
    mean_p_ed = np.sum(res['p_ed']) / num_episodes
    mean_p_Tm = np.sum(res['p_Tm']) / num_episodes
    mean_p_soc = np.sum(res['p_soc']) / num_episodes
    mean_jerk = np.mean(np.abs(res['jerk']))
    return success_rate, mean_cost, mean_p_ed, mean_p_Tm, mean_p_soc, mean_jerk


def cut_in_eval_all(save_path, model_pt, env_id, num_episodes):
    cut_in_res = {
        'succ': [],
        'cost': [],
        'p_ed': [],
        'p_Tm': [],
        'p_soc': [],
        'jerk': [],
    }
    for i in save_path:
        succ, cost, p_ed, p_Tm, p_soc, jerk = cut_in_eval(i, model_pt, 'CutIn-v1', num_episodes=100)
        for k, v in cut_in_res.items():
            v.append(locals()[k])
    for k, v_list in cut_in_res.items():
        if k == 'succ':
            formatted = [f"{v:.0%}" for v in v_list]
        else:
            formatted = [f"{v:.4f}" for v in v_list]
        print(f"{k}: [{', '.join(formatted)}]")
    print(f'\nCut-in evaluation done! '
          f'\n\tSuccess rate: {np.mean(cut_in_res["succ"]):.2%}, '
          f'\n\tMean cost: {np.mean(cut_in_res["cost"]):.2f}, '
          f'\n\tMean p_ed: {np.mean(cut_in_res["p_ed"]):.2f}, '
          f'\n\tMean p_Tm: {np.mean(cut_in_res["p_Tm"]):.2f}, '
          f'\n\tMean p_soc: {np.mean(cut_in_res["p_soc"]):.2f}, '
          f'\n\tMean jerk: {np.mean(cut_in_res["jerk"]):.3f}')


def save_model_wandb(save_dir: str, model_name: str):
    """Save the model weights and biases to mat files.
    Example:
    save_model_wandb(
        save_dir='runs/TD3AQ-{COCHTCLT-v2}/seed-005-2025-06-05-22-18-17',
        model_name='epoch-250.pt',
    )
    """

    evaluator = Evaluator()
    evaluator.load_saved(
        save_dir=save_dir,
        model_name=model_name,
    )

    # # save the traced model
    # model_path_actor = os.path.join(save_dir, 'traced_model_actor.pt')
    # model_path_critic = os.path.join(save_dir, 'traced_model_critic.pt')
    # obs = evaluator._env.reset()[0]
    # if hasattr(evaluator, '_critic'):
    #     trace_actor = torch.jit.trace(
    #         func=evaluator.traced_predict_actor,
    #         example_inputs=obs,
    #     )
    #     act_cont = evaluator.traced_predict_actor(obs)
    #     obs_act = torch.cat([obs, act_cont])
    #     trace_critic = torch.jit.trace(
    #         func=evaluator.traced_predict_critic,
    #         example_inputs=obs_act,
    #     )
    #     # trace_critic = torch.onnx.dynamo_export(
    #     #     trace_critic,
    #     #     evaluator._env.reset()[0],
    #     #     model_path_critic,
    #     # )
    #     trace_critic.save(model_path_critic)
    # else:
    #     trace_actor = torch.jit.trace(
    #         func=evaluator.traced_predict_actor,
    #         example_inputs=evaluator._env.reset()[0]
    #     )
    # trace_actor.save(model_path_actor)

    if any(algo in save_dir for algo in ['PPO', 'SAC']):
        actor_net = evaluator._actor.mean  # GaussianLearningActor
    # elif save_dir.split('/')[1] == 'TD3AQ'
    elif any(algo in save_dir for algo in ['TD3AQ', 'TD3']):
        actor_net = evaluator._actor.net  # MLPActor
    else:
        raise ValueError(f'Unsupported algorithm in {save_dir}')
    wandb_actor = {
        'fc1_weights': actor_net[0].weight.detach().cpu().numpy(),
        'fc1_bias': actor_net[0].bias.detach().cpu().numpy(),
        'fc2_weights': actor_net[2].weight.detach().cpu().numpy(),
        'fc2_bias': actor_net[2].bias.detach().cpu().numpy(),
        'fc3_weights': actor_net[4].weight.detach().cpu().numpy(),
        'fc3_bias': actor_net[4].bias.detach().cpu().numpy(),
    }
    sio.savemat(os.path.join(save_dir, 'wandb_actor.mat'), wandb_actor)
    if hasattr(evaluator, '_critic'):
        critic_net = evaluator._critic.net_lst[0][0]
        wandb_critic = {
            'fc1_weights': critic_net[0].weight.detach().cpu().numpy(),
            'fc1_bias': critic_net[0].bias.detach().cpu().numpy(),
            'fc2_weights': critic_net[2].weight.detach().cpu().numpy(),
            'fc2_bias': critic_net[2].bias.detach().cpu().numpy(),
            'fc3_weights': critic_net[4].weight.detach().cpu().numpy(),
            'fc3_bias': critic_net[4].bias.detach().cpu().numpy(),
        }
        sio.savemat(os.path.join(save_dir, 'wandb_critic.mat'), wandb_critic)
    print('Saved the weights and biases successfully!')


if __name__ == '__main__':
    # Fill your experiment's log directory in here.
    # Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48

    # for equivalent mf
    # LOG_DIR = 'runs/TD3AQ-{COCHTCLT-v2}/seed-003-2025-07-15-11-43-27'
    # LOG_DIR = 'runs/TD3AQLag-{COCHTCLT-v0}/seed-001-2025-07-15-09-59-17'
    # LOG_DIR = 'runs/TD3-{COCHTCLT-v3}/seed-004-2025-07-11-22-46-15'
    # LOG_DIR = 'runs/TD3Lag-{COCHTCLT-v1}/seed-003-2025-07-11-22-37-14'
    # LOG_DIR = 'runs/SAQ-{COCHTCLT-v2}/seed-002-2025-07-12-18-24-44' # x
    # LOG_DIR = 'runs/SAQLag-{COCHTCLT-v0}/seed-004-2025-07-12-22-07-23' # x
    # LOG_DIR = 'runs/SAC-{COCHTCLT-v3}/seed-005-2025-07-16-01-32-22'
    # LOG_DIR = 'runs/SACLag-{COCHTCLT-v1}/seed-001-2025-07-15-21-13-21'
    # LOG_DIR = 'runs/PPO-{COCHTCLT-v3}/seed-003-2025-07-12-13-05-15'  # cut-in
    # LOG_DIR = 'runs/PPO-{COCHTCLT-v3}/seed-002-2025-07-12-12-42-38'
    # LOG_DIR = 'runs/PPOLag-{COCHTCLT-v1}/seed-003-2025-07-16-11-27-37'

    # for test
    # LOG_DIR = 'runs/PPOLag-{COCHTCLT-v1}/seed-004-2025-07-30-08-07-59'
    LOG_DIR = 'runs/PPO-{COCHTCLT-v3}/seed-001-2025-08-04-20-17-13'
    # LOG_DIR = 'runs/TD3AQLag-{COCHTCLT-v0}/seed-005-2025-07-15-03-08-33'

    # for eff_Ah
    # LOG_DIR = 'runs/PPOLag-{COCHTCLT-v1}/seed-003-2025-07-19-19-36-26'  # 0
    # LOG_DIR = 'runs/PPOLag-{COCHTCLT-v1}/seed-003-2025-07-24-09-43-48'  # 0.5
    # LOG_DIR = 'runs/PPOLag-{COCHTCLT-v1}/seed-003-2025-07-16-11-27-37'  # 1
    # LOG_DIR = 'runs/PPOLag-{COCHTCLT-v1}/seed-003-2025-07-19-20-06-43'  # 2
    # LOG_DIR = 'runs/PPOLag-{COCHTCLT-v1}/seed-003-2025-07-19-20-39-28'  # 4
    # LOG_DIR = 'runs/PPOLag-{COCHTCLT-v1}/seed-003-2025-07-20-16-10-05'  # 8

    # for cut-in
    # cutin_path = [
    #     'runs/PPO-{COCHTCLT-v3}/seed-001-2025-07-12-12-19-51',
    #     'runs/PPO-{COCHTCLT-v3}/seed-002-2025-07-12-12-42-38',
    #     'runs/PPO-{COCHTCLT-v3}/seed-003-2025-07-12-13-05-15',
    #     'runs/PPO-{COCHTCLT-v3}/seed-004-2025-07-12-13-28-02',
    #     'runs/PPO-{COCHTCLT-v3}/seed-005-2025-07-12-13-50-53'
    # ]
    cutin_path = [
        'runs/PPOLag-{COCHTCLT-v1}/seed-001-2025-07-16-10-26-08',
        'runs/PPOLag-{COCHTCLT-v1}/seed-002-2025-07-16-10-56-51',
        'runs/PPOLag-{COCHTCLT-v1}/seed-003-2025-07-16-11-27-37',
        'runs/PPOLag-{COCHTCLT-v1}/seed-004-2025-07-16-11-59-29',
        'runs/PPOLag-{COCHTCLT-v1}/seed-005-2025-07-16-12-29-10'
    ]

    # # for seq-opt EMS equivalent fuel
    # LOG_DIR = 'runs/PPOLag-{SeqACCCHTCLT-v1}/seed-004-2025-11-21-11-33-24'

    MODEL_NAME = 'epoch-best.pt'

    EVAL_ENVS = [
        # # un-CMDP and CMDP can use the same env to evaluate
        # # 'COCHTCLT-v0',  # v0, HA
        # # 'COWHVC-v0',
        # # 'COHDUDDS-v0',
        # # 'COJE05-v0',
        # # 'COHHDDT-v0',
        #
        # # 'COCHTCLT-v0',  # v0, CMDP, HA (TD3AQ)
        # 'COCHTCLT-v1',  # v1, CMDP, PA

        'SeqEMSCHTCLT-v1',  # v1, CMDP, PA
    ]

    # set random seed
    seed = 1
    npr.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # equivalent_fuel_eval(LOG_DIR, MODEL_NAME, EVAL_ENVS)
    # cut_in_eval(LOG_DIR, MODEL_NAME, 'CutIn-v1')
    # cut_in_eval_all(cutin_path, MODEL_NAME, 'CutIn-v1', num_episodes=100)
    rew, cost, info = evaluation_one(LOG_DIR, MODEL_NAME, save_mat=False, eval_env='COCHTCLT')
    # evaluation_all(LOG_DIR, save_mat=False)
    # save_model_wandb(LOG_DIR, MODEL_NAME)
