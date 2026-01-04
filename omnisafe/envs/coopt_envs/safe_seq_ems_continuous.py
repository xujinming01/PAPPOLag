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
from scipy.constants import g
from scipy.interpolate import interp1d, RegularGridInterpolator

from utils import CYCLE_LENGTHS, DT, BattAging
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
l_gear = 0.05


class EMSEnv(gym.Env):
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
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
        )  # the frist 3 are gear shift, then Te

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
                # self.velo_low,
                self.acce_low,
                # self.distance_low,
                # self.error_low,
                self.soc_low,
                self.gear_idx_low,
                # self.torque_low,
            ], dtype=np.float64),
            high=np.array([
                self.velo_high,
                # self.velo_high,
                self.acce_high,
                # self.distance_high,
                # self.error_high,
                self.soc_high,
                self.gear_idx_high,
                # self.torque_high,
            ], dtype=np.float64),
            dtype=np.float64
        )

        # self.state = np.array([0, 0, 0, 0, 0])  # vh, ah, soc, gear, Tw, len=5
        self.state = np.array([0, 0, 0, 0])  # vh, ah, soc, gear, len=4
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

        self.shift_count = 0

    def step(self, action: list):
        terminated = False
        # ----------------- state -----------------
        # vr = self.state[0]  # ralative velocity
        vh = self.state[0]  # velocity of the host vehicle
        ah = self.state[1]
        soc = self.state[2]
        gear_idx_prev = int(self.state[3])
        Tw = self._gettorque(vh, ah)
        # Tw = self.state[4]  # torque demand at wheels

        # ----------------- action -----------------
        assert len(action) == 4, 'action length error'
        # # assert 0 <= action[-1] <= 1, 'engine torque error'
        # one_hot = np.argmax(action[:6])  # choose the discrete with the largest value
        # clutch, shift = self._get_discrete_action(one_hot)
        shift = int(np.argmax(action[:3]))
        Te_act = action[-1]  # engine torque, continuous

        # ----------------- EMS -----------------
        # # assert vh >= 0, f'vh < 0: {vh} on count {self.count}'
        # if -1e-8 < vh_n < 0:
        #     vh_n = 0

        # Tw = self._gettorque(vh, ah_n)

        if DT == 1 or (DT == 0.1 and self.count % (1 / DT) == 0):
            next_gear = gear_idx_prev + shift - 1  # [0, 1, 2] -> [-1, 0, 1]
            if next_gear < 0 or next_gear > 5:
                # p_gear = penalty_max  # penalty for invalid gear shift
                gear_idx = gear_idx_prev  # invalid shift, do nothing
            else:
                # p_gear = 0
                gear_idx = next_gear
        else:
            pass

        ig = GEAR[gear_idx]  # gear ratio, discrete
        w = vh / wheel_radius * ifd * ig  # w = v/r * i_d * ig
        if w > w_MAX:
            # p_w = cost_max
            p_w = 0
            while w > w_MAX:
                gear_idx += 1
                ig = GEAR[gear_idx]
                w = vh / wheel_radius * ifd * ig
            # w = w_MAX
        else:
            p_w = 0

        # gear shifting penalty
        gear_diff = abs(gear_idx - gear_idx_prev)
        self.shift_count += gear_diff

        Te_max = f_Te_max(w)
        # Te = Te_act * (Te_max - Te_idle) + Te_idle  # [Te_idle, Te_max]
        Te = Te_act * Te_max  # [0, Te_max]
        Te_org = Te

        # if not clutch or w < we_idle:  # clutch off or w < we_idle
        #     Te = Te_idle
        #     we = we_idle
        # else:  # clutch on, engine driving
        #     we = w

        if w < we_idle or Tw <= 0:  # w<we_idle or decelerating
        # if w < we_idle:
            # Te = Te_idle
            Te = 0
            # we = we_idle
            we = 0
            clutch = 0
        else:  # Te_drive > 0 then clutch engaged
            we = w
            clutch = 1

        # Te_drive = Te - Te_idle
        Te_drive = Te
        mfdot = f_mfdot([we, Te])[0]
        mf = mfdot * DT
        if Te_drive == 0:
            clutch = 0
            mf = 0
        # assert Te_drive >= 0, f'Te_drive < 0: {Te_drive} at step {self.count}'

        # calculate the motor torque
        Tm_max = min(Pb_max / w, Tm_MAX) if w > 0 else Tm_MAX
        # Tm_need = (Tw * eta_g * eta_f) / (ig * ifd) - Te_drive
        Tm_need = Tw / ((ig * ifd) * ((eta_g * eta_f) ** np.sign(Tw))) - Te_drive
        Tm = Tm_need
        Tb = 0
        p_Tm = 0
        if Tm_need < -Tm_max:
            Tm = -Tm_max
            Tb = (Te_drive + Tm) * ig * ifd - Tw * (eta_g * eta_f)
            # p_Tm = cost_max
        elif Tm_need > Tm_max:
            Tm = Tm_max
            Te_drive = Tw / (ig * eta_g * ifd * eta_f) - Tm
            # Te = Te_drive + Te_idle
            Te = Te_drive
            p_Tm = cost_max * np.tanh((Tm_need - Tm_max) / Tm_MAX)
            # assert p_Tm <= 1, f'p_Tm > 1: {p_Tm} at step {self.count}'
            if Te > Te_max:
                Te = Te_max
                p_Tm = cost_max  # cannot meet the demand
                # terminated = True
            mfdot = f_mfdot([we, Te])[0]
            mf = mfdot * DT
            clutch = 1 if Te_drive > 0 else 0

        # soc update
        eta_m = f_eta_m([w, Tm])[0]
        Pb = Tm * w * eta_m ** np.sign(-Tm)
        voc = f_voc(soc)
        rb = f_rb(soc)
        tmp = voc ** 2 - 4 * rb * Pb
        assert tmp >= 0, f'voc^2 - 4*rb*Pb < 0: {tmp}'
        bat_curr = (voc - np.sqrt(tmp)) / (2 * rb)  # battery current
        soc_dot = -bat_curr / (Qmax * 3600)  # soc change rate
        soc = soc + soc_dot * DT

        # penalty for Q-loss
        # alpha = alpha_high if soc > 0.45 else alpha_low
        # beta = beta_high if soc > 0.45 else beta_low
        # c_rate = abs(bat_curr) / Qmax
        # Q_Ah = abs(bat_curr) * DT / 3600  # Ah
        # Q_loss = (alpha * soc + beta) * np.exp((-Ea + eta * c_rate) / (R_gas * T_batt)) * (Q_Ah ** z)
        # Q_loss = min(Q_Ah, Q_loss)  # limit the loss to the passed Ah
        # r_Qloss = - Q_loss / Qloss_max  # normalized reward for battery aging
        # # r_Qloss = 0  # no battery aging

        # penalty for effective Ah-throughput
        c_rate = abs(bat_curr) / Qmax
        # severity = calculate_severity(soc, T_batt, c_rate)
        severity = BattAging.calculate_severity(soc, BattAging.T_batt, c_rate)
        eff_Ah = severity * abs(bat_curr) * DT / 3600  # effective Ah throughput
        r_Ah = - eff_Ah / Ah_max  # normalized reward for Ah throughput

        # penalty for soc deviation
        e_soc = abs(soc - 0.6)
        r_soc = - e_soc / (0.6 - soc_lower)
        # r_soc = - e_soc / 0.2 * mf_max * self.count / self.ep_len

        # r_soc = - (e_soc / 0.2) ** 2
        # r_soc = - e_soc / 0.2 * mf_max * 5 * self.count / self.ep_len
        # r_soc = 0

        if soc > soc_upper:
            p_soc = cost_max * (np.abs(soc - soc_upper) / (1 - soc_upper))
        elif soc < soc_lower:
            p_soc = cost_max * (np.abs(soc - soc_lower) / (soc_lower - 0))
        else:
            p_soc = 0
        soc = 1 if soc > 1 else 0 if soc < 0 else soc

        # with shift and clutch penalty
        # reward = float(- kf * mf  # fuel cost
        #                - ke * (Pb / (3600 * 1000) * dt)  # electricity cost
        #                - p_gear - p_clutch)

        # without shift and clutch penalty
        # reward = float(- kf * mf - ke * (Pb / (3600 * 1000) * dt))

        # pure energy cost
        # r_ems = float(- kf * mf  # fuel cost
        #               - ke * (Pb / (3600 * 1000) * dt))  # electricity cost
        currency = kf * mf + ke * (Pb / (3600 * 1000)) * DT
        c_ems = p_soc + p_w + p_Tm
        r_mf = - mf / mf_max  # fuel consumption
        p_gear = - gear_diff
        # r_ems = r_mf + r_soc * mf_max
        # r_ems = - currency

        # ----------------- reward and cost -----------------

        # coopt reward
        # reward = r_acc + r_ems
        reward = (
             # r_e * l_e
             # + r_jerk * l_jerk
             + r_mf * l_mf
             + r_soc * l_soc
             + r_Ah * l_Ah
             + p_gear * l_gear
        # ) / (l_e + l_jerk + l_mf + l_soc)
        # ) / (l_mf + l_soc + l_Ah)  # normalized rewards
        ) / (l_mf + l_soc + l_Ah + l_gear)  # normalized rewards
        # ) / (l_e + l_jerk + l_mf + l_Ah)
        # reward += 1  # step forward reward

        cost = c_ems
        p_ed = 0

        # if self.eval and (p_Tm or p_w or p_soc or p_gear or p_clutch):
        if c_ems:
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
            if p_Tm:
                # print(f'\tstep: {self.count}, p_Tm: {p_Tm}')
                self.addi_penalty['p_Tm'] += 1
            if p_w:
                # print(f'\tstep: {self.count}, p_w: {p_w}')
                self.addi_penalty['p_w'] += 1
            if p_soc:
                # print(f'\tstep: {self.count}, p_soc: {p_soc}')
                self.addi_penalty['p_soc'] += 1
            # if p_gear:
            #     # print(f'    p_gear: {p_gear}')
            #     self.addi_penalty['p_gear'] += 1
            # if p_clutch:
            #     # print(f'    p_clutch: {p_clutch}')
            #     self.addi_penalty['p_clutch'] += 1

        info = {
            'Tw': Tw, 'Te': Te, 'Tm': Tm, 'Tb': Tb, 'gear': gear_idx,
            'wm': w, 'soc': soc, 'we': we,
            'reward': reward, 'Te_org': Te_org, 'cost': cost,
            'clutch': clutch, 'shift': shift, 'currency': currency,
            'p_Tm': p_Tm, 'p_w': p_w, 'p_soc': p_soc,
            'v': vh, 'a': ah, 'jerk': 0, 'ed': 0, 'd': self.dist, 'mf': mf,
            'p_vh': 0, 'p_collision': 0, 'p_ed': p_ed,
            'r_ed': 0, 'r_jerk': 0, 'r_mf': r_mf, 'r_soc': r_soc, 'r_Ah': r_Ah,
            'vp': vh, 'ap': ah,
            'c_rate': c_rate, 'shift_count': self.shift_count,
        }

        self.count += 1
        self.ii = self.ii + 1
        vh_n = self.vi_vec[self.ii]
        ah_n = self.ai_vec[self.ii]
        Tw_n = self._gettorque(vh_n, ah_n)

        truncated = True if self.count >= self.ep_len else False
        # terminated = True if c_acc or self.count >= self.cycle_length else False
        if (terminated or truncated) and self.eval:
            print(f'\n{self.cycle} cycle')
            for k, v in self.addi_penalty.items():
                print(f'\t{k}: {v}')
        # if terminated and not truncated:
        #     reward += -mf_max * 4 * (self.ep_len - self.count)

        # new state
        # unscaled_state = [vh_n, ah_n, soc, gear_idx, Tw_n]
        unscaled_state = [vh_n, ah_n, soc, gear_idx]
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
        # self.dist = 6 + 0.8 * vh + 0.07 * vh ** 2
        # ed = 0
        soc = np.random.uniform(soc_lower, soc_upper)
        # soc = 0.6
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
        self.vp = 0

        # self.state = np.array([vh, ah, soc, gear_idx, Tw], dtype=np.float64)
        self.state = np.array([vh, ah, soc, gear_idx], dtype=np.float64)
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
        # self.ep_len = len(self.ai_vec_orig)
        # self.ai_vec = np.concatenate((self.ai_vec_orig, self.ai_vec_orig), axis=0)
        # self.vi_vec = np.concatenate((self.vi_vec_orig, self.vi_vec_orig), axis=0)

        # self.state = np.array([0, 0, soc_init, 0, 0], dtype=np.float64)
        self.state = np.array([0, 0, soc_init, 0], dtype=np.float64)
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

    def _cut_in(self, vh):
        pass


class CHTCLTEMSEnv(EMSEnv):

    def __init__(self):
        super(CHTCLTEMSEnv, self).__init__("chtc_lt")


class HDUDDSEMSEnv(EMSEnv):

    def __init__(self):
        super(HDUDDSEMSEnv, self).__init__("hd_udds")


class JE05EMSEnv(EMSEnv):

    def __init__(self):
        super(JE05EMSEnv, self).__init__("je05")


class WHVCEMSEnv(EMSEnv):

    def __init__(self):
        super(WHVCEMSEnv, self).__init__("whvc")


class WHVC3000EMSEnv(EMSEnv):

    def __init__(self):
        super(WHVC3000EMSEnv, self).__init__("whvc3000")


class HHDDTEMSEnv(EMSEnv):

    def __init__(self):
        super(HHDDTEMSEnv, self).__init__("hhddt")


class CutInEnv(EMSEnv):
    """Cut-in environment"""

    def __init__(self):
        super(CutInEnv, self).__init__("chtc_lt")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset the state with random initial velocity."""
        super().reset(seed=seed)

        # Generate a preceding vehicle with constant velocity.
        v_min, v_max = 60 / 3.6, 90 / 3.6  # 16.67 ~ 27.78 m/s
        # v_min, v_max = 50 / 3.6, 85 / 3.6  # *0.85, 13.89 ~ 23.61 m/s
        vp = np.random.uniform(v_min, v_max)  # random constant speed
        # vp = 90 / 3.6

        assert DT == 0.1, 'CutInEnv only supports DT=0.1'
        self.ep_len = 500
        self.vi_vec = np.full(self.ep_len, vp)  # velocity vector (constant)
        self.ai_vec = np.zeros(self.ep_len)  # acceleration vector (zero)

        # host vehicle initial state
        self.ii = 0
        vh = self.vi_vec[self.ii]
        ah = self.ai_vec[self.ii]
        self.dist = 6 + 0.8 * vh + 0.07 * vh ** 2
        soc = 0.6
        gear_idx = 4 if vp < 70 / 3.6 else 5  # [60, 70), 5, [70, 100], 6
        clutch = 1

        self.vp = vp
        self.state = np.array([vh, ah, self.dist, soc, gear_idx], dtype=np.float64)
        # self.state = np.array([vh, ah, 0, soc, gear_idx], dtype=np.float64)
        self.count = 0
        self.eval = False
        self._is_cut_in = True  # enable cut-in action

        for key in self.addi_penalty.keys():
            self.addi_penalty[key] = 0

        if options is not None:
            if 'eval' in options:
                self._evaluate_on()

        info = {
            'Tw': 0, 'Te': 0, 'Tm': 0, 'Tb': 0, 'gear': gear_idx,
            'w': 0, 'soc': soc,
            'reward': 0, 'Tm_org': 0, 'Te_org': 0,
            'clutch': clutch, 'shift': 0, 'currency': 0,
            'cost': 0, 'p_Tm': 0, 'p_w': 0, 'p_soc': 0,
            'v': vh, 'a': ah, 'jerk': 0, 'ed': 0, 'd': self.dist,
            'mf': 0, 'p_vh': 0, 'p_collision': 0, 'p_ed': 0,
            'r_ed': 0, 'r_j': 0, 'r_mf': 0, 'r_soc': 0, 'r_Ah': 0,
            'vp': vp, 'ap': 0,
            'c_rate': 0,
        }

        return self.state, info

    def _cut_in(self, vh):
        """Cut-in action, change the preceding vehicle's velocity"""
        # if self.vp <= 60 / 3.6:
        #     vp = self.vp / 2
        #     ttc = 4
        # elif self.vp <= 80 / 3.6:
        #     # vp = 30 / 3.6
        #     vp = self.vp / 2
        #     ttc = 4
        # elif self.vp <= 100 / 3.6:
        #     vp = 40 / 3.6
        #     # vp = self.vp / 2
        #     ttc = 5
        # else:
        #     vp = 50 / 3.6
        #     ttc = 6
        # self.dist = (vh - vp) * ttc
        # self.vi_vec = np.full(self.ep_len, vp)

        dist_opt = 6 + 0.8 * vh + 0.07 * vh ** 2
        dist_min = 2 + 0.4 * vh + 0.06 * vh ** 2
        self.dist = (dist_opt + dist_min) / 2
        # self.dist = (self.dist + dist_min) / 2
        vp = vh * 0.75
        self.vi_vec = np.full(self.ep_len, vp)
