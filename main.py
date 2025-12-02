import os
import time

import numpy as np
import pandas as pd
import scipy.io as sio
from line_profiler import profile

import omnisafe
# from omnisafe.envs.coopt_envs.constants import TRAIN_LEN, DT
from plot_coopt import plot_acc, plot_ems
from utils import CYCLES, DT


@profile
def main(algorithm, environment_id, cfgs):
    agent = omnisafe.Agent(algorithm, environment_id, custom_cfgs=cfgs)
    agent.learn()

    agent.plot(
        smooth=1,  # smooth=0 means no smooth
    )  # default TestEpRet and TestEpCost, if not, EpRet and EpCost

    ep_rew, ep_cost, best_model = agent.evaluate(num_episodes=1)

    # plot the ACC results
    log_dir = agent.agent.logger.log_dir
    model_dir = os.path.join(log_dir, 'eval_mat', best_model + '.mat')
    mat_rl = sio.loadmat(model_dir)
    # mat_cycle = sio.loadmat('matlab/cycles/whvc_1s.mat')
    mat_cycle = sio.loadmat('matlab/cycles/chtc_lt_1s.mat')
    len_cyc, len_rl = len(mat_cycle['vp'][0]), len(mat_rl['v'][0])
    len_xt = min(len_cyc, len_rl) - 1
    xt = np.arange(0, len_xt * DT, DT)
    plot_acc(mat_cycle, mat_rl, xt, best_model, save_to=log_dir + '/fig-acc.png')

    # plot the EMS results
    # mat_ref = sio.loadmat('mpc/coopt_whvc_20250412_1122_rl.mat')
    mat_ref = sio.loadmat('mpc/coopt_chtc_lt_20250426_1716_rl.mat')
    # mat_ref = sio.loadmat('matlab/results_hev_simplified_noTk_1whvc.mat')
    res_ems = {
        'Te': [mat_rl['Te'][0], mat_ref['Te'][0]],
        'Tm': [mat_rl['Tm'][0], mat_ref['Tm'][0]],
        # 'Tb': [mat_rl['Tb'][0], mat_ref['Tb'][0]],
        'SOC': [mat_rl['soc'][0], mat_ref['soc'][0]],
        # 'Clutch': [mat_rl['clutch'][0], mat_ref['clutch'][0]],
        'Gear': [mat_rl['gear'][0] + 1, mat_ref['gear'][0] + 1],  # mpc need +1
    }
    plot_ems(res_ems, xt, best_model, save_to=log_dir + '/fig-ems.png')

    return ep_rew, ep_cost, best_model, mat_rl


if __name__ == '__main__':
    assert DT == 1, "DT must be 1 second for now, please change it in utils.py"
    total_epoch = 1000
    save_model_freq = 1000
    start_time = time.time()
    # v0, cd,safe; v1, c, safe; v2, cd, penalty; v3, c, penalty
    # cfg = {'algo': 'SAQLag', 'env': 'COCHTCLT-v0'}
    # cfg = {'algo': 'TD3AQLag', 'env': 'COCHTCLT-v0'}

    cfg = {'algo': 'PPOLag', 'env': 'COCHTCLT-v1'}
    # cfg = {'algo': 'TD3Lag', 'env': 'COCHTCLT-v1'}
    # cfg = {'algo': 'SACLag', 'env': 'COCHTCLT-v1'}
    # cfg = {'algo': 'SACPID', 'env': 'COCHTCLT-v1'}
    # cfg = {'algo': 'CUP', 'env': 'COCHTCLT-v1'}

    # cfg = {'algo': 'TD3AQ', 'env': 'COCHTCLT-v2'}
    # cfg = {'algo': 'SAQ', 'env': 'COCHTCLT-v2'}

    # cfg = {'algo': 'PPO', 'env': 'COCHTCLT-v3'}
    # cfg = {'algo': 'TD3', 'env': 'COCHTCLT-v3'}
    # cfg = {'algo': 'SAC', 'env': 'COCHTCLT-v3'}

    env = cfg['env'].split('-')[0]
    cycle_length = CYCLES[env]['length']
    # cycle_length = TRAIN_LEN
    total_steps = cycle_length * total_epoch

    PPO_cfgs = {
        'train_cfgs': {
            'total_steps': total_steps,
            'device': 'cpu',
            'torch_threads': 1,
            'vector_env_nums': 1,
            'parallel': 1,  # for GPUs, start from device number
        },
        'algo_cfgs': {
            'steps_per_epoch': cycle_length,
            'update_iters': 10,
            'batch_size': 256,
            'entropy_coef': 0.01,
            'gamma': 0.995,
            'lam': 0.98,
            'lam_c': 0.98,
            'clip': 0.2,
            'obs_normalize': False,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': False,
            'save_model_freq': save_model_freq,
        },
        'model_cfgs': {
            'exploration_noise_anneal': False,
            'std_range': [0.5, 0.1],
            'actor': {
                'hidden_sizes': [64, 64],
                'lr': 3e-4,
            },
            'critic': {
                'hidden_sizes': [64, 64],
                'lr': 3e-4,
            },
        },
    }
    PPOS_lag_cfgs = {
        'lagrange_cfgs': {
            'cost_limit': 0.0,
            'lagrangian_multiplier_init': 0.001,
            'lambda_lr': 0.035,
        },
    }

    algo_cfgs = {
        'PPO': PPO_cfgs,
        'PPOLag': PPO_cfgs | PPOS_lag_cfgs,
        'CUP': PPO_cfgs | PPOS_lag_cfgs,
        'FOCOPS': PPO_cfgs | PPOS_lag_cfgs,
        'TD3': {
            'train_cfgs': {
                'total_steps': total_steps,
                'device': 'cpu',
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,  # for GPUs, start from device number
            },
            'algo_cfgs': {
                'steps_per_epoch': cycle_length,
                'update_iters': 1,
                'start_learning_steps': -1,  # -1 means start learning immediately NOTE to remove
                'gamma': 0.99,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'use_tensorboard': False,
                'save_model_freq': save_model_freq,
            },
            'model_cfgs': {
                'actor': {
                    'lr': 0.0003,
                    'hidden_sizes': [64, 64],
                },
                'critic': {
                    'lr': 0.0003,
                    'hidden_sizes': [64, 64],
                },
            },
        },
        'TD3Lag': {
            'train_cfgs': {
                'total_steps': total_steps,
                'device': 'cpu',
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,  # for GPUs, start from device number
            },
            'algo_cfgs': {
                'steps_per_epoch': cycle_length,
                'update_iters': 1,
                'start_learning_steps': -1,
                'gamma': 0.99,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'use_tensorboard': False,
                'save_model_freq': save_model_freq,
            },
            'model_cfgs': {
                'actor': {
                    'lr': 0.0003,
                    'hidden_sizes': [64, 64],
                },
                'critic': {
                    'lr': 0.0003,
                    'hidden_sizes': [64, 64],
                },
            },
            'lagrange_cfgs': {
                'cost_limit': 0.0,
                'lagrangian_multiplier_init': 0.001,
                'lambda_lr': 0.00001,
            },
        },
        'SAC': {
            'train_cfgs': {
                'total_steps': total_steps,
                'device': 'cpu',
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,  # for GPUs, start from device number
            },
            'algo_cfgs': {
                'steps_per_epoch': cycle_length,
                'update_iters': 1,
                'start_learning_steps': -1,
                'gamma': 0.995,
                # 'auto_alpha': True,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'use_tensorboard': False,
                'save_model_freq': save_model_freq,
            },
            'model_cfgs': {
                'actor': {
                    'lr': 3e-4,
                    'hidden_sizes': [64, 64],
                },
                'critic': {
                    'lr': 3e-4,
                    'hidden_sizes': [64, 64],
                },
            },
        },
        'SAQ': {
            'train_cfgs': {
                'total_steps': total_steps,
                'device': 'cpu',
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,  # for GPUs, start from device number
            },
            'algo_cfgs': {
                'steps_per_epoch': cycle_length,
                'update_iters': 1,
                'start_learning_steps': -1,  # -1 means start learning immediately
                'gamma': 0.995,
                'batch_size': 256,
                'auto_alpha': True,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'use_tensorboard': False,
                'save_model_freq': save_model_freq,
            },
            'model_cfgs': {
                'actor': {
                    'lr': 3e-4,
                    'hidden_sizes': [64, 64],
                },
                'critic': {
                    'lr': 1e-3,
                    'hidden_sizes': [64, 64],
                },
            },
        },
        'SACLag': {
            'train_cfgs': {
                'total_steps': total_steps,
                'device': 'cpu',
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,  # for GPUs, start from device number
            },
            'algo_cfgs': {
                'steps_per_epoch': cycle_length,
                'update_iters': 1,
                'warmup_epochs': 1,
                'start_learning_steps': -1,
                'gamma': 0.995,
                # 'obs_normalize': True,
                'batch_size': 256,
                'alpha': 0.2,  # initial alpha value
                'auto_alpha': True,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'use_tensorboard': False,
                'save_model_freq': save_model_freq,  # do not save model during training
            },
            'model_cfgs': {
                # 'linear_lr_decay': True,
                'actor': {
                    'lr': 3e-4,
                    'hidden_sizes': [64, 64],
                },
                'critic': {
                    'lr': 3e-4,
                    'hidden_sizes': [64, 64],
                },
            },
            'lagrange_cfgs': {
                'cost_limit': 0.0,
                'lagrangian_multiplier_init': 0.001,
                'lambda_lr': 0.00001,
            },
        },
        'SAQLag': {
            'train_cfgs': {
                'total_steps': total_steps,
                'device': 'cpu',
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,  # for GPUs, start from device number
            },
            'algo_cfgs': {
                'steps_per_epoch': cycle_length,
                'update_iters': 1,
                'batch_size': 256,
                'start_learning_steps': -1,  # -1 means start learning immediately
                'gamma': 0.995,
                # 'warmup_epochs': 20,
                # 'auto_alpha': True,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'use_tensorboard': False,
                'save_model_freq': save_model_freq,
            },
            'model_cfgs': {
                'actor': {
                    'lr': 5e-5,
                    'hidden_sizes': [64, 64],
                },
                'critic': {
                    'lr': 5e-5,
                    'hidden_sizes': [64, 64],
                },
            },
            'lagrange_cfgs': {
                'cost_limit': 0.0,
                'lagrangian_multiplier_init': 0.001,
                'lambda_lr': 0.00001,
            },
        },
        'SACPID': {
            'train_cfgs': {
                'total_steps': total_steps,
                'device': 'cpu',
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,  # for GPUs, start from device number
            },
            'algo_cfgs': {
                'steps_per_epoch': cycle_length,
                'update_iters': 1,
                'warmup_epochs': 1,
                'start_learning_steps': -1,
                'gamma': 0.995,
                # 'obs_normalize': True,
                'batch_size': 256,
                'auto_alpha': True,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'use_tensorboard': False,
                'save_model_freq': save_model_freq,
            },
            'model_cfgs': {
                'actor': {
                    'lr': 3e-4,
                    'hidden_sizes': [64, 64],
                },
                'critic': {
                    'lr': 1e-3,
                    'hidden_sizes': [64, 64],
                },
            },
            'lagrange_cfgs': {
                'cost_limit': 0.0,
                'lagrangian_multiplier_init': 0.001,
                'pid_kp': 0.000001,
                'pid_ki': 0.0000001,
                'pid_kd': 0.0000001,
                'pid_d_delay': 10,
                'pid_delta_p_ema_alpha': 0.95,
                'pid_delta_d_ema_alpha': 0.95,
            },
        },
        # 'CPPOPID': {
        #     'train_cfgs': {
        #         'total_steps': total_steps,
        #         'device': 'cpu',
        #         'torch_threads': 32,
        #         'parallel': 1,  # for GPUs, start from device number
        #     },
        #     'algo_cfgs': {
        #         'steps_per_epoch': cycle_length,
        #         'update_iters': 1,
        #         'lam': 0.95,
        #         'lam_c': 0.95,
        #     },
        #     'logger_cfgs': {
        #         'use_wandb': False,
        #         'use_tensorboard': False,
        #         'save_model_freq': save_model_freq,
        #     },
        #     'model_cfgs': {
        #         'actor': {
        #             'hidden_sizes': [64, 64],
        #             'lr': 0.0003,
        #         },
        #         'critic': {
        #             'hidden_sizes': [64, 64],
        #             'lr': 0.0003,
        #         },
        #     },
        #     'lagrange_cfgs': {
        #         'cost_limit': 0.0,
        #         'lagrangian_multiplier_init': 0.001,
        #         'pid_kp': 0.1,
        #         'pid_ki': 0.01,
        #         'pid_kd': 0.01,
        #         'pid_d_delay': 10,
        #         'pid_delta_p_ema_alpha': 0.95,
        #         'pid_delta_d_ema_alpha': 0.95,
        #     },
        # },
        'TD3AQ': {
            'train_cfgs': {
                'total_steps': total_steps,
                'device': 'cpu',
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,  # for GPUs, start from device number
            },
            'algo_cfgs': {
                'steps_per_epoch': cycle_length,
                'update_iters': 1,
                'start_learning_steps': -1,  # -1 means start learning immediately
                # 'gamma': 0.9801256558248345,
                'gamma': 0.995,
                # 'obs_normalize': True,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'use_tensorboard': False,
                'save_model_freq': save_model_freq,
            },
            'model_cfgs': {
                'actor': {
                    # 'lr': 0.0021502881652993074,
                    'lr': 0.0003,
                    'hidden_sizes': [64, 64],
                },
                'critic': {
                    # 'lr': 0.0020700573615610616,
                    'lr': 0.001,
                    'hidden_sizes': [64, 64],
                },
            },
        },
        'TD3AQLag': {
            'train_cfgs': {
                'total_steps': total_steps,
                'device': 'cpu',
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,  # for GPUs, start from device number
            },
            'algo_cfgs': {
                'steps_per_epoch': cycle_length,
                'update_iters': 1,
                'start_learning_steps': -1,  # -1 means start learning immediately
                'warmup_epochs': 20,
                'gamma': 0.995,
                'obs_normalize': False,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'use_tensorboard': False,
                'save_model_freq': save_model_freq,
            },
            'model_cfgs': {
                'linear_lr_decay': False,
                'actor': {
                    'lr': 0.0003,
                    'hidden_sizes': [64, 64],
                },
                'critic': {
                    'lr': 0.001,
                    'hidden_sizes': [64, 64],
                },
            },
            'lagrange_cfgs': {
                'cost_limit': 0.0,
                'lagrangian_multiplier_init': 0.001,
                'lambda_lr': 0.00001,
            },
        }
    }

    custom_cfgs = algo_cfgs[cfg['algo']]

    # different seeds
    # seeds = [1, 2, 3, 4, 5]
    # results = {seed: {'best_r': None, 'best_c': None} for seed in seeds}
    # for seed in seeds:
    #     custom_cfgs['seed'] = seed
    #     best_rew, best_cost, best_md = main(cfg['algo'], cfg['env'], custom_cfgs)
    #     results[seed]['best_r'] = best_rew
    #     results[seed]['best_c'] = best_cost
    # # # print(f'run {algo} on {env_id} with seeds {seeds} done!')
    # # print(f'best_rews: {best_rs}')
    # # # print(f'best gaps: {gaps}')
    # # print(f'best_costs: {best_cs}')
    # # # print(f'gap: {np.mean(gaps):.2%} +- {np.std(gaps):.2%}')
    # # print(f'rews: {np.mean(best_rs):.2f} +- {np.std(best_rs):.2f}')
    # # print(f'costs: {np.mean(best_cs):.2f} +- {np.std(best_cs):.2f}')
    # # # print(f'best models: {best_mds}')
    #
    # print(f'\nResults for {cfg["algo"]} on {cfg["env"]}:')
    # for seed, res in results.items():
    #     print(f'Seed {seed}: Best Reward: {res["best_r"]:.2f}, Best Cost: {res["best_c"]:.2f}')
    # print(f'Average Reward: {np.mean([res["best_r"] for res in results.values()]):.2f}')
    # print(f'Average Cost: {np.mean([res["best_c"] for res in results.values()]):.2f}')

    # seeds = [1, 2, 3, 4, 5]
    seeds = [0]  # for debugging, use a single seed
    # seeds = [7, 8, 9]  # for debugging, multiple seeds
    records = []
    for seed in seeds:
        custom_cfgs['seed'] = seed
        best_rew, best_cost, best_md, res_mat = main(cfg['algo'], cfg['env'], custom_cfgs)
        records.append({'seed': seed, 'best_r': best_rew, 'best_c': best_cost, 'res_mat': res_mat})
    df = pd.DataFrame(records)
    print(f'\nResults for {cfg["algo"]}-{{{cfg["env"]}}}:')
    for _, row in df.iterrows():
        print(f'Seed {int(row["seed"])}: Best Reward: {row["best_r"]:.2f}, Best Cost: {row["best_c"]:.2f}')
    r_mean, r_std = df['best_r'].mean(), df['best_r'].std()
    c_mean, c_std = df['best_c'].mean(), df['best_c'].std()
    print(f'Average Reward and cost: {r_mean:.2f}±{r_std:.2f}, {c_mean:.2f}±{c_std:.2f}')
    # sort to find the best seed (min cost, max reward)
    best_row = df.sort_values(by=['best_c', 'best_r'], ascending=[True, False]).iloc[0]
    print(f'\nBest Seed: {int(best_row["seed"])} (Reward and cost: {best_row["best_r"]:.2f}, {best_row["best_c"]:.2f})')
    best_res_mat = best_row['res_mat']
    print(f'Fuel consumption [kg]: {np.sum(best_res_mat["mf"][0]):.3f}')
    print(f'Final SOC: {best_res_mat["soc"][0][-1]:.4f}')
    print(f'Mean of |jerk| [m/s³]: {np.abs(best_res_mat["jerk"][0]).mean():.3f}')
    print(f'Mean of |ed| [m]: {np.abs(best_res_mat["ed"][0]).mean():.3f}')
    print(f'p_ed: {np.sum(best_res_mat["p_ed"][0])}')
    print(f'p_w: {np.sum(best_res_mat["p_w"][0])}')
    print(f'p_Tm: {np.sum(best_res_mat["p_Tm"][0])}')
    print(f'p_soc: {np.sum(best_res_mat["p_soc"][0])}')
    print('r_jerk | r_ed | r_mf | r_soc | r_Ah:')
    print(f'{np.sum(best_res_mat["r_jerk"][0]):.2f} | '
          f'{np.sum(best_res_mat["r_ed"][0]):.2f} | '
          f'{np.sum(best_res_mat["r_mf"][0]):.2f} | '
          f'{np.sum(best_res_mat["r_soc"][0]):.2f} | '
          f'{np.sum(best_res_mat["r_Ah"][0]):.2f}')


    # # single seed
    # custom_cfgs['seed'] = 5
    # best_rew, best_cost, best_md = main(cfg['algo'], cfg['env'], custom_cfgs)
    # # print(f'Gap: {- best_rew / CYCLES[env]["dp_optimal"] - 1:.2%}')

    print(f'\nTraining time: {(time.time() - start_time) / 3600:.2f}h')
