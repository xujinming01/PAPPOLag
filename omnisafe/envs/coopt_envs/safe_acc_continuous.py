"""
IDD using Safety Gymnasium API
plug-in,
continuous action: engine torque
discrete action: gear shift, clutch
"""

from typing import Optional

import gymnasium as gym
import numpy as np
import scipy.io as sio
import torch
from scipy.constants import g
from scipy.interpolate import interp1d, RegularGridInterpolator

from utils import CYCLE_LENGTHS, DT, BattAging
from omnisafe.evaluator import Evaluator

DT_SUFF = {1: '1', 0.1: '0dot1'}

# constants

# DT = 1  # time step

w_MAX = 250  # max engine/motor angular velocity
Te_MAX = 600  # max engine torque
Tm_MAX = 600  # max motor torque
Tb_min = -6000  # min mechanical brake torque
Tmb_min = -6000  # min wheel torque

ifd = 4.11  # final drive gear ratio, another choice is 3.55
eta_f = 0.931  # final drive efficiency
eta_g = 0.931  # transmission efficiency

Qmax = 26  # battery capacity, Ah
soc_upper = 0.8
soc_lower = 0.4
# soc_init = 0.6  # initial battery state of charge
Ic_max = 10
I_max = Ic_max * Qmax  # max current, A
Ah_max = I_max * DT / 3600  # max Ah throughout

m = 5000  # vehicle mass
wheel_radius = 0.5715  # wheel radius
Cd = 0.65  # air drag coefficient
rou = 1.1985  # air density
A = 6.73  # frontal area
mu = 0.01  # rolling resistance coefficient
theta = 0  # road slope

ALPHA = 10000  # coefficient for fuel consumption

Te_idle = 25  # engine idle torque, Nm
we_idle = 100  # engine idle angular velocity, rad/s

ke = 1  # electricity price CNY/kWh
kf = 9.0357  # fuel price CNY/kg

# fuel rate interpolation
x_we = np.arange(100, 275, 25)
y_Te = np.arange(0, 700, 100)
z_mfdot = np.array([
    [0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001],
    [0.0007, 0.0009, 0.001, 0.0013, 0.0015, 0.0018, 0.002],
    [0.00125, 0.0015, 0.0018, 0.0021, 0.0025, 0.0028, 0.0032],
    [0.00175, 0.0021, 0.0025, 0.0029, 0.0033, 0.004, 0.0045],
    [0.0023, 0.0028, 0.0031, 0.0037, 0.0043, 0.0052, 0.0058],
    [0.003, 0.0035, 0.004, 0.0046, 0.0055, 0.0064, 0.007],
    [0.0037, 0.0043, 0.005, 0.006, 0.007, 0.008, 0.009]
])
mfdot_max = 0.01  # max fuel rate
mf_max = mfdot_max * DT
# mf_max = 1
cost_max = 1  # for CMDP, reward and cost are separate, 1 for easy count
f_mfdot = RegularGridInterpolator((x_we, y_Te), z_mfdot.T, 'linear', bounds_error=False, fill_value=mfdot_max)

# max engine torque interpolation
y_Te_max = np.array([420, 510, 600, 575, 550, 500, 480])
# x_we = np.array([100, 150, 250])
# y_Te_max = np.array([420, 600, 480])
f_Te_max = interp1d(x_we, y_Te_max, 'linear', fill_value="extrapolate")

# motor efficiency interpolation
x_wm = np.arange(0, 275, 25)
y_Tm = np.arange(-600, 700, 100)
z_eta_m = np.array([
    [0.725, 0.7, 0.75, 0.825, 0.85, 0.875, 0.9, 0.9, 0.9, 0.875, 0.875],
    [0.725, 0.7, 0.775, 0.85, 0.875, 0.9, 0.92, 0.9, 0.9, 0.875, 0.875],
    [0.775, 0.71, 0.825, 0.875, 0.89, 0.9, 0.91, 0.91, 0.9, 0.875, 0.875],
    [0.79, 0.75, 0.825, 0.875, 0.89, 0.9, 0.925, 0.925, 0.875, 0.825, 0.85],
    [0.8, 0.75, 0.825, 0.875, 0.89, 0.9, 0.91, 0.9, 0.85, 0.775, 0.75],
    [0.825, 0.8, 0.85, 0.87, 0.875, 0.875, 0.875, 0.825, 0.775, 0.7, 0.7],
    [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
    [0.825, 0.8, 0.85, 0.87, 0.875, 0.875, 0.875, 0.825, 0.775, 0.7, 0.7],
    [0.8, 0.75, 0.825, 0.875, 0.89, 0.9, 0.91, 0.9, 0.85, 0.775, 0.75],
    [0.79, 0.75, 0.825, 0.875, 0.89, 0.9, 0.925, 0.925, 0.875, 0.825, 0.85],
    [0.775, 0.71, 0.825, 0.875, 0.89, 0.9, 0.91, 0.91, 0.9, 0.875, 0.875],
    [0.725, 0.7, 0.775, 0.85, 0.875, 0.9, 0.92, 0.9, 0.9, 0.875, 0.875],
    [0.725, 0.7, 0.75, 0.825, 0.85, 0.875, 0.9, 0.9, 0.9, 0.875, 0.875]
])
f_eta_m = RegularGridInterpolator((x_wm, y_Tm), z_eta_m.T, 'linear', bounds_error=False, fill_value=0.7)
Tm_TOLE = 0.01  # tolerance for motor torque

# max motor torque interpolation
# y_Tm_max = np.array([600, 600, 600, 600, 600, 600, 600, 600, 500, 330, 280])
# # x_wm = np.array([0, 175, 250])
# # y_Tm_max = np.array([600, 600, 280])
# f_Tm_max = interp1d(x_wm, y_Tm_max, 'linear')  # no extrapolation
Pb_max = 90000  # use Pb_max to calculate the max motor torque

# soc interpolation
x_soc = np.linspace(0, 1, num=11, endpoint=True)
y_voc = np.array(
    [3.57, 3.69, 3.74, 3.76, 3.78, 3.8, 3.86, 3.92, 4, 4.1, 4.21]) * 112
f_voc = interp1d(x_soc, y_voc, 'linear')

# battery resistance interpolation
y_rb = np.array(
    [2, 1.5, 1.34, 1.31, 1.32, 1.3, 1.28, 1.31, 1.3, 1.31, 1.32]) * 0.001 * 112
# x_soc = np.array([0, 0.2, 1])
# y_rb = np.array([2, 1.31, 1.31]) * 0.001 * 112
f_rb = interp1d(x_soc, y_rb, 'linear')

# discrete actions
GEAR = {
    0: 6.25,
    1: 3.583,
    2: 2.22,
    3: 1.36,
    4: 1,
    5: 0.74,
}


ed_MAX = 25  # max distance error, 4 + 0.4 * 30 + 0.01 * 30 ** 2 = 25, ed_opt2min
ed_NORM = 20  # max counted ed, 3.2 + 0.32 * 30 + 0.008 * 30 ** 2 = 20, ed_inf2min
vr_MAX = 3  # max relative velocity
ah_MAX = 2.5  # max acceleration
jerk_MAX = ah_MAX / DT  # max jerk

l_e = 3
l_jerk = 3
l_mf = 1
l_soc = 1
l_Ah = 1

class ACCEnv(gym.Env):
    def __init__(self, cycle):
        # import drive cycles
        self.cycle = cycle
        self.mat_mgs = sio.loadmat(f'matlab/cycles/{self.cycle}_{DT_SUFF[DT]}s.mat')
        self.ai_vec_orig = self.mat_mgs['ap'][0]
        self.vi_vec_orig = self.mat_mgs['vp'][0]
        TRAIN_LEN = CYCLE_LENGTHS[self.cycle]
        self.ep_len = TRAIN_LEN
        self.ai_vec, self.vi_vec = [], []
        # self.action_space = gym.spaces.Box(
        #     low=0.,
        #     high=1.,
        #     shape=(8,),
        #     # the first 6 are discrete one-hot, then Te, Tm
        # )
        self.action_space = gym.spaces.Box(
            low=np.array([-ah_MAX], dtype=np.float32),
            high=np.array([ah_MAX], dtype=np.float32),
        )  # ah

        # observation space: vehicle velocity, torque demand at wheels, soc
        self.velo_low = 0
        self.velo_high = 30
        self.acce_low = -ah_MAX
        self.acce_high = ah_MAX
        self.distance_low = 0
        self.distance_high = 120
        self.error_low = -ed_MAX
        self.error_high = ed_MAX
        self.torque_low = -10000
        self.torque_high = 10000
        self.soc_low = 0
        self.soc_high = 1
        self.gear_idx_low = 0
        self.gear_idx_high = 5
        # self.clutch_low = 0
        # self.clutch_high = 1
        self.Te_low = 0
        self.Te_high = Te_MAX
        # self.Tmb_low = Tmb_min
        # self.Tmb_high = Te_max
        self.observation_space = gym.spaces.Box(
            low=np.array([
                self.velo_low,
                self.velo_low,
                self.acce_low,
                self.distance_low,
                self.error_low,
                # self.soc_low,
                # self.gear_idx_low,
                # self.clutch_low,
            ], dtype=np.float64),
            high=np.array([
                self.velo_high,
                self.velo_high,
                self.acce_high,
                self.distance_high,
                self.error_high,
                # self.soc_high,
                # self.gear_idx_high,
                # self.clutch_high,
            ], dtype=np.float64),
            dtype=np.float64
        )

        self.state = np.array([0, 0, 0, 0, 0])  # vr, vh, ah, distance, e
        self.ii = 0  # index of drive cycle
        self.count = 0  # count of steps in an episode
        self.eval = False  # evaluation mode
        self.addi_penalty = {
            # 'p_vh': 0,
            # 'p_collision': 0,
            # 'p_vr': 0,
            'p_ed': 0,
            'p_Tm': 0,
            'p_w': 0,
            # 'p_gear': 0,
            # 'p_clutch': 0,
            'p_soc': 0,
        }

        # ACC data
        self.max_jerk = 0
        self.vr_max = 5
        self.dist = 0
        # proceeding vehicle
        self.vp = 0

        # cut-in flag
        self._is_cut_in = False

        # # EMS model related
        # self._ems_state = [0, 0, 0, 0, 0]
        # self._ems_model = self._load_model()
        #
        # self.shift_count = 0

    def step(self, action: list):
        terminated = False
        # ----------------- state -----------------
        vr = self.state[0]  # ralative velocity
        vh = self.state[1]  # velocity of the host vehicle
        ah = self.state[2]
        dist = self.state[3]
        ed = self.state[4]
        # soc = self.state[5]
        # gear_idx = self.state[6]
        # clutch_state = self.state[7]

        # ----------------- action -----------------
        # assert len(action) == 8, 'action length error'
        # # assert 0 <= action[-1] <= 1, 'engine torque error'
        # one_hot = np.argmax(action[:6])  # choose the discrete with the largest value
        # clutch, shift = self._get_discrete_action(one_hot)
        # shift = np.argmax(action[:3])
        # Te_act = action[-2]  # engine torque, continuous
        ah_n = action[-1]  # acceleration, continuous

        # ----------------- ACC -----------------
        # constraint for backward driving
        ah_n_min = - vh / DT
        ah_n = ah_n_min if ah_n < ah_n_min else ah_n

        vp = self.vp
        vp_n = self.vi_vec[self.ii]
        ap_n = (vp_n - vp) / DT

        vh_n = vh + ah_n * DT
        vr_n = vp_n - vh_n
        vr_n_esti = vp - vh_n

        # tiny value filter
        if vh_n < 1e-8:
            vh_n = 0
            vr_n = vp_n

        # # constraint for jerk
        # jerk_abs = abs((ah_n - ah) / dt)
        # if jerk_abs > 1:
        #     if ah_n < ah:
        #         ah_n = ah - 0.01 * jerk_abs  # reduce jerk to 0.1 original
        #     else:
        #         ah_n = ah + 0.01 * jerk_abs
        #
        # # constraint for distance error
        # # a_opt = (dist + 0.2 * vp_n - 0.95725 * vh_n - 6) / 0.80725
        # # if abs(ed) > 1 and (ah_n < a_opt - 0.2 or ah_n > a_opt + 0.2):
        # #     ah_n = a_opt
        # if abs(ed) > 1e-4:
        #     if ed < 0:  # too close, need to decelerate
        #         ah_n = np.clip(0.7 * ah + 0.55 * ed, self.acce_low, 0)
        #     else:
        #         ah_n = np.clip(0.7 * ah + 0.55 * ed, 0, self.acce_high)

        # # constraint for start and stop driving
        # ah_rd = np.random.uniform(-0.05, 0)
        # if vp <= 0.5 and (vh_n < 0.1 and ah <= 0):
        #     ah_n = - vh_n / dt  # stop for low speed
        # if vp > 0.05 and (vh_n < 0.1 and ah > 0) and (ah_n < (0.02 - ah_rd) / dt):
        #     ah_n = (0.02 - ah_rd) / dt
        # if vp > 0.05 and (0.1 < vh_n < 0.5 and ah > 0) and (ah_n < (0.04 - ah_rd) / dt):
        #     ah_n = (0.05 + ah_rd) / dt

        # # constraint for collision
        # if dist < 4.5:
        #     ah_n = - 1.5 + ah_rd

        # optimal distance
        dist_opt = 6 + 0.8 * vh_n + 0.07 * vh_n ** 2

        dist_min = 2 + 0.4 * vh_n + 0.06 * vh_n ** 2
        dist_max = 10 + 1.2 * vh_n + 0.08 * vh_n ** 2
        dist_inf = 5.2 + 0.72 * vh_n + 0.068 * vh_n ** 2
        dist_sup = 6.8 + 0.88 * vh_n + 0.072 * vh_n ** 2
        ed_opt2inf = dist_opt - dist_inf
        ed_opt2min = dist_opt - dist_min

        # # distance update
        # if self._is_cut_in and self.count == 250:
        #     self._cut_in(vh_n)
        # else:
        self.dist += (vp + vp_n) / 2 * DT - (vh + vh_n) / 2 * DT

        # distance error, absolute dist to dist_opt
        ed = self.dist - dist_opt
        # if vh == 0:
        #     ed = 0  # no error if zero velocity

        # jerk
        jerk = (ah_n - ah) / DT
        # if jerk > self.max_jerk:
        #     self.max_jerk = jerk

        # penalty
        # p_vh = cost_max if vh_n < -1e-8 else 0  # backward driving
        # p_collision = cost_max if dist <= 0 else 0  # collision
        # # p_vr = cost_max if abs(vr_n_esti) > self.vr_max else 0  # relative velocity
        # p_ed = cost_max if abs(ed) > ed_MAX else 0  # distance error

        # if vh_n < -1e-8:  # backward driving
        #     p_vh = cost_max
        #     ah_n = - vh / dt
        #     vh_n = 0
        # else:
        #     p_vh = 0
        # assert vh_n >= 0, f'vh_n < 0: {vh_n} at step {self.count}'
        p_vh = 0

        # # reset the velocity and position if collsion or too far
        # if self.dist <= 0:  # collision
        #     p_collision = cost_max * 2
        #     # ah_n = (vp_n - vp) / dt
        #     vh_n = vp_n
        #     self.dist = dist_opt
        #     # self.dist = self.dist + 1
        # else:
        #     p_collision = 0
        p_collision = 0

        if abs(ed) > ed_opt2min:  # ed out of bound
        # if ed > ed_opt2min:  # too far, close use p_colli to control
            p_ed = cost_max
            if not self._is_cut_in:
                # # ah_n = (vp_n - vp) / DT
                # vh_n = vp_n
                # self.dist = dist_opt

                # ah_need = (vp_n - vh) / DT
                # ah_n = np.clip(ah_need, -ah_MAX, ah_MAX)
                ah_n = np.clip(ed * 1, max(ah_n_min, -ah_MAX), ah_MAX)
                jerk = (ah_n - ah) / DT
                vh_n = vh + ah_n * DT

            # terminated = True
        else:
            p_ed = 0

        # reward and cost
        # r_e = - (ed / 1) ** 2
        # r_vr = - (vr_n_esti / 5) ** 2
        # r_jerk = - (jerk / 5) ** 2
        # r_ah = - (ah_n / 3) ** 2

        # # linear penalty in opt range, quadratic penalty out of range
        # if abs(ed) <= ed_opt2inf:
        #     r_e = - abs(ed / ed_opt2inf)
        # else:
        #     r_e = (ed / ed_opt2inf) ** 2

        # no penalty in opt range, penalty in inf-min range
        if abs(ed) <= ed_opt2inf:
            # r_e = (ed_opt2inf - abs(ed)) / ed_MAX
            r_e = 0
        else:
            # r_e = - abs(ed / ed_opt2min)  # linear
            r_e = - (abs(ed) - ed_opt2inf) / ed_NORM
            # r_e = - abs(ed) / ed_MAX
            # r_e = (ed / ed_opt2min) ** 2  # quadratic

        # # penalty consistent in min-max range
        # r_e = - abs(ed / ed_opt2min)
        # # r_e = 0

        # r_vr = - abs(vr_n / vr_MAX)
        r_vr = 0
        # r_ah = - abs(ah_n / ah_MAX)
        r_ah = 0
        r_jerk = - abs(jerk / jerk_MAX)

        # # coordinate with MPC
        # r_e = - (ed / ed_MAX) ** 2
        # r_vr = - (vr_n / vr_MAX) ** 2
        # r_ah = - (ah_n / ah_MAX) ** 2
        # r_jerk = - (jerk / jerk_MAX) ** 2

        c_acc = p_vh + p_collision + p_ed
        # r_acc = (r_e + r_jerk + r_vr + r_ah) * mf_max
        # r_acc = (r_e + r_jerk + r_ah + 3) * mf_max

        # ----------------- reward and cost -----------------

        # coopt reward
        # reward = r_acc + r_ems
        reward = (
            r_e * l_e
            + r_jerk * l_jerk
        ) / (l_e + l_jerk)
        cost = c_acc

        # if self.eval and (p_Tm or p_w or p_soc or p_gear or p_clutch):
        if c_acc:
            # print(f'{self.count} steps')
            # if p_vh:
            #     # print(f'    p_vh: {p_vh}')
            #     self.addi_penalty['p_vh'] += 1
            # if p_collision:
            #     # print(f'    p_collision: {p_collision}')
            #     self.addi_penalty['p_collision'] += 1
            # if p_vr:
            #     # print(f'    p_vr: {p_vr}')
            #     self.addi_penalty['p_vr'] += 1
            if p_ed:
                # print(f'\tstep: {self.count}, p_ed: {p_ed}')
                self.addi_penalty['p_ed'] += 1
            # if p_Tm:
            #     # print(f'\tstep: {self.count}, p_Tm: {p_Tm}')
            #     self.addi_penalty['p_Tm'] += 1
            # if p_w:
            #     # print(f'\tstep: {self.count}, p_w: {p_w}')
            #     self.addi_penalty['p_w'] += 1
            # if p_soc:
            #     # print(f'\tstep: {self.count}, p_soc: {p_soc}')
            #     self.addi_penalty['p_soc'] += 1
            # if p_gear:
            #     # print(f'    p_gear: {p_gear}')
            #     self.addi_penalty['p_gear'] += 1
            # if p_clutch:
            #     # print(f'    p_clutch: {p_clutch}')
            #     self.addi_penalty['p_clutch'] += 1

        info = {
            'Tw': 0, 'Te': 0, 'Tm': 0, 'Tb': 0, 'gear': 0,
            'wm': 0, 'soc': 0, 'we': 0,
            'reward': reward, 'Te_org': 0, 'cost': cost,
            'clutch': 0, 'shift': 0, 'currency': 0,
            'p_Tm': 0, 'p_w': 0, 'p_soc': 0,
            'v': vh_n, 'a': ah_n, 'jerk': jerk, 'ed': ed, 'd': self.dist, 'mf': 0,
            'p_vh': p_vh, 'p_collision': p_collision, 'p_ed': p_ed,
            'r_ed': r_e, 'r_jerk': r_jerk, 'r_mf': 0, 'r_soc': 0, 'r_Ah': 0,
            'vp': vp_n, 'ap': ap_n,
            'c_rate': 0,  'shift_count': 0,
        }

        self.count += 1
        self.ii = self.ii + 1
        self.vp = vp_n
        truncated = True if self.count >= self.ep_len else False
        # terminated = True if c_acc or self.count >= self.cycle_length else False
        if (terminated or truncated) and self.eval:
            print(f'\n{self.cycle} cycle')
            for k, v in self.addi_penalty.items():
                print(f'\t{k}: {v}')
        # if terminated and not truncated:
        #     reward += -mf_max * 4 * (self.ep_len - self.count)

        # new state
        unscaled_state = [vr_n, vh_n, ah_n, self.dist, ed]
        self.state[:] = unscaled_state

        return self.state, reward, cost, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset the state with random initial values"""
        super().reset(seed=seed)

        # random initialization on zero velocity
        # zero_indices = np.where(self.vi_vec_orig == 0)[0]
        zero_indices = np.where(self.vi_vec_orig[:self.ep_len] == 0)[0]
        self.ii = np.random.choice(zero_indices)
        self.ai_vec = np.concatenate((self.ai_vec_orig[:self.ep_len], self.ai_vec_orig[:self.ep_len]), axis=0)
        self.vi_vec = np.concatenate((self.vi_vec_orig[:self.ep_len], self.vi_vec_orig[:self.ep_len]), axis=0)
        vr = 0
        vh = self.vi_vec[self.ii]
        ah = self.ai_vec[self.ii]
        assert vh == 0, f'vh != 0: {vh} on count {self.count}'
        Tw = self._gettorque(vh, ah)
        self.dist = 6 + 0.8 * vh + 0.07 * vh ** 2
        ed = 0
        # soc = np.random.uniform(soc_lower, soc_upper)
        soc = 0.6
        gear_idx = 0
        clutch = 0
        # shift_cd = 10

        # # no random initialization
        # self.ii = 0
        # vr = 0
        # vh = 0
        # ah = 0
        # Tw = 0
        # dist = 6  # (Li2017, Jian Song)
        # e = 0
        # # soc = np.random.uniform(soc_lower, soc_upper)
        # soc = 0.6
        # gear_idx = 0
        # clutch = 0

        self.max_jerk = 0
        self.state = np.array([vr, vh, ah, self.dist, ed], dtype=np.float64)
        self.count = 0
        self.eval = False

        for key in self.addi_penalty.keys():
            self.addi_penalty[key] = 0

        if options is not None:
            if 'reset_zero_soc' in options:
                self._reset_zero(options['reset_zero_soc'])
            if 'eval' in options:
                self._evaluate_on()

        info = {
            'Tw': Tw, 'Te': 0, 'Tm': 0, 'Tb': 0, 'gear': gear_idx,
            'w': 0, 'soc': soc,
            'reward': 0, 'Tm_org': 0, 'Te_org': 0,
            'clutch': clutch, 'shift': 0, 'currency': 0,
            'cost': 0, 'p_Tm': 0, 'p_w': 0, 'p_soc': 0,
            'v': vh, 'a': ah, 'jerk': 0, 'ed': 0, 'd': self.dist,
            'mf': 0, 'p_vh': 0, 'p_collision': 0, 'p_ed': 0,
            'r_ed': 0, 'r_j': 0, 'r_mf': 0, 'r_soc': 0, 'r_Ah': 0,
            'vp': ah, 'ap': ah,
            'c_rate': 0,
        }

        # np.random.seed(seed)

        return self.state, info

    def _reset_zero(self, soc_init):
        """Reset the state with zero initial values"""
        self.state = np.array([0, 0, 0, 6, 0], dtype=np.float64)
        self.count = 0
        self.ii = 0

        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    @staticmethod
    def _gettorque(vi, ai) -> float:
        """Calculate torque demand at wheels"""
        # if vi and ai == 0:
        #     return 0
        # else:
        #     return wheel_radius * (m * ai
        #                            + 0.5 * Cd * rou * A * vi ** 2
        #                            + mu * m * g * np.cos(theta)
        #                            + m * g * np.sin(theta))
        return wheel_radius * (m * ai
                               + 0.5 * Cd * rou * A * vi ** 2
                               + mu * m * g * np.cos(theta) * np.tanh(4 * vi)
                               + m * g * np.sin(theta))

    @staticmethod
    def _getacce(Tw, vi) -> float:
        """Calculate acceleration, remove rolling coefficient judgment"""
        # a = (
        #     Tw / wheel_radius
        #     - 0.5 * Cd * rou * A * vi ** 2
        #     - mu * m * g * np.cos(theta)
        #     - m * g * np.sin(theta)
        # ) / m
        # if vi == 0 and a < 0:
        #     return 0  # no backward driving
        # else:
        #     return a
        return (
            Tw / wheel_radius
            - 0.5 * Cd * rou * A * vi ** 2
            - mu * m * g * np.cos(theta) * np.tanh(4 * vi)
            # - mu * m * g * np.cos(theta) * np.clip(2.5 * vi, 0, 1)
            - m * g * np.sin(theta)
        ) / m

    def _evaluate_on(self):
        """Set evaluation mode, print penalty during evaluation"""
        self.eval = True

    @staticmethod
    def _get_discrete_action(action):
        """Get shift and clutch action from one-hot vector
        For one-hot, the first half is clutch off, the second half is clutch on
        """
        clutch = action // 3
        shift = action % 3

        return clutch, shift


class CHTCLTACCEnv(ACCEnv):

    def __init__(self):
        super(CHTCLTACCEnv, self).__init__("chtc_lt")


class HDUDDSACCEnv(ACCEnv):

    def __init__(self):
        super(HDUDDSACCEnv, self).__init__("hd_udds")


class JE05ACCEnv(ACCEnv):

    def __init__(self):
        super(JE05ACCEnv, self).__init__("je05")


class WHVCACCEnv(ACCEnv):

    def __init__(self):
        super(WHVCACCEnv, self).__init__("whvc")


class WHVC3000ACCEnv(ACCEnv):

    def __init__(self):
        super(WHVC3000ACCEnv, self).__init__("whvc3000")


class HHDDTACCEnv(ACCEnv):

    def __init__(self):
        super(HHDDTACCEnv, self).__init__("hhddt")
