"""
IDD using Safety Gymnasium API
continuous action: engine torque
discrete action: gear shift, clutch
"""

from typing import Optional

import gymnasium as gym
import numpy as np
import scipy.io as sio
from scipy.constants import g
from scipy.interpolate import interp1d, interp2d

from utils import CYCLE_LENGTHS, DT

# constants

# DT = 1  # time step

w_max = 250  # max engine/motor angular velocity
Te_max = 600  # max engine torque
Tb_min = -6000  # min mechanical brake torque
Tmb_min = -6000  # min wheel torque

ifd = 4.11  # final drive gear ratio, another choice is 3.55
eta_f = 0.931  # final drive efficiency
eta_g = 0.931  # transmission efficiency

Qmax = 6.5 * 3600  # battery capacity
soc_upper = 0.8
soc_lower = 0.4
# soc_init = 0.6  # initial battery state of charge

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
mfdot_max = 0.005  # max fuel rate
mf_max = mfdot_max * DT
cost_max = 1  # for CMDP, reward and cost are separate, 1 for easy count
f_mfdot = interp2d(x_we, y_Te, z_mfdot, 'linear', fill_value=mfdot_max)

# max engine torque interpolation
# y_Te_max = np.array([420, 510, 600, 575, 550, 500, 480])
x_we = np.array([100, 150, 250])
y_Te_max = np.array([420, 600, 480])
f_Te_max = interp1d(x_we, y_Te_max, 'linear', fill_value="extrapolate")

# motor efficiency interpolation
# x_wm = np.arange(0, 275, 25)
# y_Tm = np.arange(-600, 700, 100)
# z_eta_m = np.array([
#     [0.725, 0.7, 0.75, 0.825, 0.85, 0.875, 0.9, 0.9, 0.9, 0.875, 0.875],
#     [0.725, 0.7, 0.775, 0.85, 0.875, 0.9, 0.92, 0.9, 0.9, 0.875, 0.875],
#     [0.775, 0.71, 0.825, 0.875, 0.89, 0.9, 0.91, 0.91, 0.9, 0.875, 0.875],
#     [0.79, 0.75, 0.825, 0.875, 0.89, 0.9, 0.925, 0.925, 0.875, 0.825, 0.85],
#     [0.8, 0.75, 0.825, 0.875, 0.89, 0.9, 0.91, 0.9, 0.85, 0.775, 0.75],
#     [0.825, 0.8, 0.85, 0.87, 0.875, 0.875, 0.875, 0.825, 0.775, 0.7, 0.7],
#     [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
#     [0.825, 0.8, 0.85, 0.87, 0.875, 0.875, 0.875, 0.825, 0.775, 0.7, 0.7],
#     [0.8, 0.75, 0.825, 0.875, 0.89, 0.9, 0.91, 0.9, 0.85, 0.775, 0.75],
#     [0.79, 0.75, 0.825, 0.875, 0.89, 0.9, 0.925, 0.925, 0.875, 0.825, 0.85],
#     [0.775, 0.71, 0.825, 0.875, 0.89, 0.9, 0.91, 0.91, 0.9, 0.875, 0.875],
#     [0.725, 0.7, 0.775, 0.85, 0.875, 0.9, 0.92, 0.9, 0.9, 0.875, 0.875],
#     [0.725, 0.7, 0.75, 0.825, 0.85, 0.875, 0.9, 0.9, 0.9, 0.875, 0.875]
# ])
# f_eta_m = interp2d(x_wm, y_Tm, z_eta_m, 'linear', fill_value=0.7)
Tm_TOLE = 0.01  # tolerance for motor torque

# max motor torque interpolation
# y_Tm_max = np.array([600, 600, 600, 600, 600, 600, 600, 600, 500, 330, 280])
x_wm = np.array([0, 175, 250])
y_Tm_max = np.array([600, 600, 280])
f_Tm_max = interp1d(x_wm, y_Tm_max, 'linear')  # no extrapolation

# soc interpolation
# x_soc = np.linspace(0, 1, num=11, endpoint=True)
# y_voc = np.array(
#     [3.57, 3.69, 3.74, 3.76, 3.78, 3.8, 3.86, 3.92, 4, 4.1, 4.21]) * 112
# f_voc = interp1d(x_soc, y_voc, 'linear')

# battery resistance interpolation
# y_rb = np.array(
#     [2, 1.5, 1.34, 1.31, 1.32, 1.3, 1.28, 1.31, 1.3, 1.31, 1.32]) * 0.001 * 112
x_soc = np.array([0, 0.2, 1])
y_rb = np.array([2, 1.31, 1.31]) * 0.001 * 112
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


ed_MAX = 5  # max distance error
vr_MAX = 3  # max relative velocity
jerk_MAX = 40  # max jerk
ah_MAX = 10  # max acceleration


class EMSEnv(gym.Env):
    def __init__(self, cycle):
        # import drive cycles
        self.cycle = cycle
        self.mat_mgs = sio.loadmat(f'matlab/cycles/{self.cycle}_{DT}s.mat')
        self.ai_vec_orig = self.mat_mgs['ap'][0]
        self.vi_vec_orig = self.mat_mgs['vp'][0]
        TRAIN_LEN = CYCLE_LENGTHS[self.cycle]
        self.ep_len = TRAIN_LEN
        self.ai_vec = np.concatenate((self.ai_vec_orig[:TRAIN_LEN], self.ai_vec_orig[:TRAIN_LEN]), axis=0)
        self.vi_vec = np.concatenate((self.vi_vec_orig[:TRAIN_LEN], self.vi_vec_orig[:TRAIN_LEN]), axis=0)

        # self.action_space = gym.spaces.Tuple((
        #     gym.spaces.Discrete(6),  # gear shift and clutch on/off
        #     gym.spaces.Box(low=0, high=1, shape=(2,)),  # Te, Tmb
        # ))
        # self.action_space = gym.spaces.Box(
        #     low=0.,
        #     high=1.,
        #     shape=(5,),
        #     # the first 3 are shift discrete, then Te, Tmb
        # )
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(3),  # gear shift
            gym.spaces.Box(low=0, high=1, shape=(2,)),  # Te, Tm
        ))

        # observation space: vehicle velocity, torque demand at wheels, soc
        self.velo_low = 0
        self.velo_high = 30
        self.acce_low = -10
        self.acce_high = 10
        self.distance_low = -10
        self.distance_high = 200
        self.error_low = -15
        self.error_high = 15
        self.torque_low = -10000
        self.torque_high = 10000
        self.soc_low = 0
        self.soc_high = 1
        self.gear_idx_low = 0
        self.gear_idx_high = 5
        # self.clutch_low = 0
        # self.clutch_high = 1
        self.Te_low = 0
        self.Te_high = Te_max
        # self.Tmb_low = Tmb_min
        # self.Tmb_high = Te_max
        self.observation_space = gym.spaces.Box(
            low=np.array([
                self.velo_low,
                self.velo_low,
                self.acce_low,
                self.distance_low,
                self.error_low,
                self.soc_low,
                self.gear_idx_low,
                # self.clutch_low,
                # self.Te_low,
                # self.Tmb_low,
                # self.acce_low,
                # 0,  # shift cd
                # 0,  # journey remaining
                # -Te_max,  # Tm_max = Te_max
            ], dtype=np.float64),
            high=np.array([
                self.velo_high,
                self.velo_high,
                self.acce_high,
                self.distance_high,
                self.error_high,
                self.soc_high,
                self.gear_idx_high,
                # self.clutch_high,
                # self.Te_high,
                # self.Tmb_high,
                # self.acce_high,
                # 10,  # shift cd
                # self.ep_len,  # journey remaining
                # Te_max,  # Tm_max = Te_max
            ], dtype=np.float64),
            dtype=np.float64
        )

        # self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # vr, vh, ah, distance, e, soc, gear, clutch, len=8
        # self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # vr, vh, ah, distance, e, soc, gear, clutch, Te, Tmb, len=10
        # self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # vp, vh, ah, distance, e, soc, gear, clutch, Te, Tmb, ap, len=11
        # self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # vp, vh, ah, distance, e, soc, gear, clutch, Te, ap, len=10
        # self.state = np.array([0, 0, 0, 0, 0, 0, 0, 10, self.ep_len, 0])  # vr, vh, ah, distance, e, soc, gear, shift_cd, ep_len, Tm, len=10
        self.state = np.array([0, 0, 0, 0, 0, 0, 0])  # vr, vh, ah, d, e, soc, gear, len=7
        self.ii = 0  # index of drive cycle
        self.count = 0  # count of steps in an episode
        self.eval = False  # evaluation mode
        self.addi_penalty = {
            'p_vh': 0,
            'p_collision': 0,
            # 'p_vr': 0,
            'p_ed': 0,
            'p_w': 0,
            # 'p_gear': 0,
            # 'p_clutch': 0,
            'p_soc': 0,
        }

        # ACC data
        self.max_jerk = 0
        self.vr_max = 5

    def step(self, action: list):
        # ----------------- state -----------------
        # vr = self.state[0]  # ralative velocity
        vh = self.state[1]  # velocity of the host vehicle
        ah = self.state[2]
        dist = self.state[3]
        ed = self.state[4]
        soc = self.state[5]
        gear_idx = self.state[6]
        # shift_cd = self.state[7]
        # Tm_last = self.state[9]

        # clutch_state = self.state[7]
        # vh = self.state[0]
        # ah = self.state[1]
        # dist = self.state[2]
        # soc = self.state[3]
        # gear_idx = self.state[4]

        # ----------------- action -----------------
        shift = action[0]
        # clutch, shift = self._get_discrete_action(action[0])
        # shift = np.argmax(action[:3])  # choose the discrete with the largest value

        Te_act = action[-2]  # engine torque, continuous
        # Tmb_act = action[-1]  # motor and brake torque after transmission, continuous
        Tm_act = action[-1]  # motor torque, continuous

        # ----------------- EMS -----------------
        # assert vh >= 0, f'vh < 0: {vh} on count {self.count}'
        # if -1e-8 < vh_n < 0:
        #     vh_n = 0
        # Tw = self._gettorque(vh_n, ah_n)

        if gear_idx + shift - 1 < 0 or gear_idx + shift - 1 > 5:
            # p_gear = penalty_max  # penalty for invalid gear shift
            pass  # invalid shift, do nothing
        else:
            # p_gear = 0
            gear_idx += shift - 1  # [0, 1, 2] -> [-1, 0, 1]

        # if Tw == 0:
        #     gear_idx = 0
        # gear_diff = abs(gear_idx - self.state[6])

        # # penalty for gear shift
        # p_gear = mf_max / 50 if gear_diff > 0 else 0
        #
        # # penalty for clutch transition
        # p_clutch = mf_max / 25 if clutch != clutch_state else 0

        ig = GEAR[gear_idx]  # gear ratio, discrete
        w = vh / wheel_radius * ifd * ig  # w = v/r * i_d * ig
        # assert w >= 0, f'w < 0: {w} on count {self.count}'
        if w > w_max:
            p_w = mf_max * 10
            while w > w_max:
                gear_idx += 1
                ig = GEAR[gear_idx]
                w = vh / wheel_radius * ifd * ig
        else:
            p_w = 0

        Te = Te_act * (f_Te_max(w) - Te_idle) + Te_idle  # [Te_idle, Te_max]
        Te_org = Te

        # if not clutch or w < we_idle:  # clutch off or w < we_idle
        #     Te = Te_idle
        #     we = we_idle
        # else:  # clutch on, engine driving
        #     we = w

        if w < we_idle:
            Te = Te_idle
            we = we_idle
            clutch = 0
        else:  # Te_drive > 0 then clutch engaged
            we = w
            clutch = 1

        Te_drive = Te - Te_idle
        mfdot = float(f_mfdot(we, Te))
        # mfdot = (
        #     0.001197
        #     - 1.01e-05 * we - 2.928e-06 * Te
        #     + 5.156e-08 * we * Te
        #     + 3.837e-08 * we ** 2 + 4.804e-09 * Te ** 2
        # )
        mf = mfdot * DT
        if Te_drive == 0:
            clutch = 0
            mf = 6.2940e-04 * DT
        # assert Te_drive >= 0, f'Te_drive < 0: {Te_drive} at step {self.count}'

        # # separate motor and brake torque
        # # Tm = 0.5 if abs(Tm - 0.5) <= Tm_TOLE else Tm  # tolerance for motor torque
        # Tm = (Tm - 0.5) / 0.5 * f_Tm_max(w)
        # Tb = Tb * Tb_min

        # # Tmb
        # if Tmb_act >= 0.5:
        #     Tmb = (Tmb_act - 0.5) / 0.5 * (f_Tm_max(w) * (ig * eta_g * ifd * eta_f))
        #     Tm = (Tmb_act - 0.5) / 0.5 * f_Tm_max(w)
        #     Tb = 0
        # else:
        #     Tmb = (0.5 - Tmb_act) / 0.5 * Tmb_min
        #     Tm = max(Tmb / (ig * eta_g * ifd * eta_f), -f_Tm_max(w))
        #     Tb = Tmb - Tm * (ig * eta_g * ifd * eta_f)
        #     # assert Tb >= Tb_min, f'Tb < Tb_min: {Tb}'

        # no Tb
        Tm = (Tm_act - 0.5) / 0.5 * f_Tm_max(w)  # Tm directly
        # Tm = np.clip(Tm_last + (Tm_act - 0.5) / 0.5 * 400, -f_Tm_max(w), f_Tm_max(w))  # Tm_last + \Delta Tm

        # Tw = (Tm + Te_drive) * ig * eta_g * ifd * eta_f + Tb
        Tw = (Tm + Te_drive) * ig * eta_g * ifd * eta_f

        # cannot backward drive constraint
        ah_n_min = - vh / DT  # min acceleration
        Tw_min = self._gettorque(vh, ah_n_min)
        if vh < 1e-2 and Tw < self._gettorque(vh, 0):  # vehicle stopped
            Tw = Tw_min
            Tm = 0
            # Tb = 0
            # Tmb = 0
        elif Tw < Tw_min:
            Tw = Tw_min

            # # Tmb
            # Tmb_before_gear = Tw / (ig * eta_g * ifd * eta_f) - Te_drive
            # # assert Tmb_before_gear <= 0, f'Tmb > 0: {Tmb_before_gear}'
            # Tm = max(Tmb_before_gear, -f_Tm_max(w))
            # Tb = (Tmb_before_gear - Tm) * (ig * eta_g * ifd * eta_f)
            # Tmb = Tb + Tm * (ig * eta_g * ifd * eta_f)

            # no Tb
            Tm = Tw / (ig * eta_g * ifd * eta_f) - Te_drive
            assert Tm >= -f_Tm_max(w), f'Tm < -f_Tm_max(w): {Tm}'
        Tb = 0

        # # accelerator and brake constraint TODO: need to check conservation of torque
        # if Te_drive > 0:  # no brake if engine driving
        #     Tb = 0
        #     Tmb = Tm * (ig * eta_g * ifd * eta_f)
        #     Tw = Te_drive * (ig * eta_g * ifd * eta_f) + Tmb

        # soc update
        # eta_m = f_eta_m(w, Tm)
        # Pb = Tm * w * eta_m ** np.sign(-Tm)
        Pb = (
            -1208
            + 1.483 * w + 0.5168 * Tm
            + 0.1686 * w ** 2 + 1.006 * w * Tm + 0.02186 * Tm ** 2
        )
        # voc = f_voc(soc)
        voc = (0.5473 * soc + 3.584) * 112
        rb = f_rb(soc)
        tmp = voc ** 2 - 4 * rb * Pb
        assert tmp >= 0, f'voc^2 - 4*rb*Pb < 0: {tmp}'
        soc_dot = -(voc - np.sqrt(tmp)) / (2 * rb * Qmax)
        soc = float(soc + soc_dot * DT)

        # penalty for soc deviation
        e_soc = abs(soc - 0.6)
        r_soc = - e_soc / 0.2 * mf_max * 5
        # r_soc = - e_soc / 0.2 * mf_max * 5 * self.count / self.ep_len
        # r_soc = 0

        if soc > soc_upper:
            p_soc = (mf_max * 10) * (np.abs(soc-soc_upper) / (1 - soc_upper))
        elif soc < soc_lower:
            p_soc = (mf_max * 10) * (np.abs(soc-soc_lower) / (soc_lower - 0))
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
        c_ems = p_soc + p_w
        r_ems = float(- mf + r_soc - c_ems)
        currency = float(kf * mf + ke * (Pb / (3600 * 1000)) * DT)

        # ----------------- ACC -----------------
        vp = self.vi_vec[self.ii]
        vp_n = self.vi_vec[self.ii + 1]
        ap_n = (vp_n - vp) / DT

        ah_n = self._getacce(Tw, vh)
        vh_n = vh + ah_n * DT
        vr_n = vp_n - vh_n
        vr_n_esti = vp - vh_n

        # tiny value filter
        if vh_n < 1e-8:
            vh_n = 0
            vr_n = vp_n

        # break_count = 462
        # if break_count <= self.count <= break_count + 5:
        #     print('here')

        # # constraint for jerk
        # jerk_abs = abs((ah_n - ah) / dt)
        # if jerk_abs > 1:
        #     if ah_n < ah:
        #         ah_n = ah - 0.01 * jerk_abs  # reduce jerk to 0.1 original
        #     else:
        #         ah_n = ah + 0.01 * jerk_abs

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
        dist_opt = 6 + 0.75 * vh + 0.0725 * vh ** 2

        # distance update
        dist_n = dist + (vp + vp_n) / 2 * DT - (vh + vh_n) / 2 * DT

        # distance error
        ed = dist_n - dist_opt
        # if vh == 0:
        #     ed = 0  # no error if zero velocity

        # jerk
        jerk = (ah_n - ah) / DT
        if jerk > self.max_jerk:
            self.max_jerk = jerk

        # penalty
        # p_vh = mf_max * 10 if vh_n < -1e-8 else 0  # backward driving
        # p_collision = mf_max * 10 if dist <= 0 else 0  # collision
        # # p_vr = cost_max if abs(vr_n_esti) > self.vr_max else 0  # relative velocity
        # p_ed = mf_max * 10 if abs(ed) > ed_MAX else 0  # distance error

        # if vh_n < -1e-8:  # backward driving
        #     p_vh = mf_max * 10
        #     ah_n = - vh / dt
        #     vh_n = 0
        # else:
        #     p_vh = 0
        # assert vh_n >= 0, f'vh_n < 0: {vh_n} at step {self.count}'
        p_vh = 0

        if dist_n <= 0:  # collision
            p_collision = mf_max * 10 * 2
            # ah_n = (vp_n - vp) / dt
            vh_n = vp_n
            dist_n = 6 + 0.75 * vh_n + 0.0725 * vh_n ** 2
            # dist_n = dist_n + 1
        else:
            p_collision = 0

        if abs(ed) > ed_MAX * 2:  # distance too far
            p_ed = mf_max * 10 * 2
            # ah_n = (vp_n - vp) / dt
            vh_n = vp_n
            dist_n = 6 + 0.75 * vh_n + 0.0725 * vh_n ** 2
            # dist_n = dist_n - 2
        else:
            p_ed = 0

        # reward and cost
        # r_e = - (ed / 1) ** 2
        # r_vr = - (vr_n_esti / 5) ** 2
        # r_jerk = - (jerk / 5) ** 2
        # r_ah = - (ah_n / 3) ** 2

        # if dist <= 0.5 * dist_opt:
        #     r_e = - abs(ed / ed_MAX) * 10  # penalty for too close
        # else:
        #     r_e = - abs(ed / ed_MAX)
        r_e = - abs(ed / ed_MAX)
        r_vr = - abs(vr_n / vr_MAX)
        r_jerk = - abs(jerk / jerk_MAX)
        # r_ah = - abs(ah_n / ah_MAX)

        c_acc = p_vh + p_collision + p_ed
        # if type(r_e) or type(r_jerk) or type(r_vr) or type(c_acc) is not (float or int):
        #     print('here')
        r_acc = (r_e + r_jerk + r_vr) * mf_max - c_acc
        # r_acc = (r_e + r_jerk + r_ah + 3) * mf_max - c_acc

        # ----------------- reward and cost -----------------

        # coopt reward
        reward = r_acc + r_ems

        if c_acc or c_ems:
            cost = (
                bool(p_vh)
                + bool(p_collision)
                + bool(p_ed)
                + bool(p_w)
                + bool(p_soc)
            )
        else:
            cost = 0

        # if self.eval and (p_Tm or p_w or p_soc or p_gear or p_clutch):
        if c_acc or c_ems:
            # print(f'{self.count} steps')
            if p_vh:
                # print(f'    p_vh: {p_vh}')
                self.addi_penalty['p_vh'] += 1
            if p_collision:
                # print(f'    p_collision: {p_collision}')
                self.addi_penalty['p_collision'] += 1
            # if p_vr:
            #     # print(f'    p_vr: {p_vr}')
            #     self.addi_penalty['p_vr'] += 1
            if p_ed:
                # print(f'    p_ed: {p_ed}')
                self.addi_penalty['p_ed'] += 1
            if p_w:
                # print(f'    p_w: {p_w}')
                self.addi_penalty['p_w'] += 1
            if p_soc:
                # print(f'    p_soc: {p_soc}')
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
            'reward': reward, 'Te_org': Te_org,
            'clutch': clutch, 'shift': shift, 'currency': currency,
            'cost': cost, 'p_w': p_w, 'p_soc': p_soc,
            'v': vh_n, 'a': ah_n, 'jerk': jerk, 'ed': ed, 'd': dist_n,
            'mf': mf, 'p_vh': p_vh, 'p_collision': p_collision, 'p_ed': p_ed,
        }

        # new state
        # unscaled_state = [vr_n, vh_n, ah_n, dist_n, ed, soc, gear_idx, clutch]
        # unscaled_state = [vr_n, vh_n, ah_n, dist_n, ed, soc, gear_idx, clutch, Te, Tmb]
        # unscaled_state = [vp_n, vh_n, ah_n, dist_n, ed, soc, gear_idx, clutch, Te, Tmb, ap_n]
        # unscaled_state = [vp_n, vh_n, ah_n, dist_n, ed, soc, gear_idx, clutch, Te, ap_n]
        # unscaled_state = [vr_n, vh_n, ah_n, dist_n, ed, soc, gear_idx, shift_cd, self.ep_len - self.count, Tm]
        unscaled_state = [vr_n, vh_n, ah_n, dist_n, ed, soc, gear_idx]
        self.state[:] = unscaled_state

        self.count += 1
        self.ii = self.ii + 1
        done = True if self.count >= self.ep_len else False
        # done = True if c_acc or self.count >= self.cycle_length else False
        if done and self.eval:
            print(f'\n{self.cycle} cycle')
            for k, v in self.addi_penalty.items():
                print(f'\t{k}: {v}')

        # for Gymnasium API
        terminated = done
        truncated = False

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
        vr = 0
        vh = self.vi_vec[self.ii]
        ah = self.ai_vec[self.ii]
        assert vh == 0, f'vh != 0: {vh} on count {self.count}'
        Tw = self._gettorque(vh, ah)
        dist = 6 + 0.75 * vh + 0.0725 * vh ** 2
        e = 0
        # soc = np.random.uniform(soc_lower, soc_upper)
        soc = 0.6
        gear_idx = 0
        clutch = 0
        shift_cd = 10

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
        # self.state = np.array([vr, vh, ah, dist, e, soc, gear_idx, clutch], dtype=np.float64)
        # self.state = np.array([vr, vh, ah, dist, e, soc, gear_idx, clutch, 0, 0], dtype=np.float64)
        # self.state = np.array([0, vh, ah, dist, e, soc, gear_idx, clutch, 0, 0, ah], dtype=np.float64)
        # self.state = np.array([0, vh, ah, dist, e, soc, gear_idx, clutch, 0, ah], dtype=np.float64)
        # self.state = np.array([vr, vh, ah, dist, e, soc, gear_idx, shift_cd, self.ep_len, 0], dtype=np.float64)
        self.state = np.array([vr, vh, ah, dist, e, soc, gear_idx], dtype=np.float64)
        self.count = 0
        self.eval = False

        info = {
            'Tw': Tw, 'Te': 0, 'Tm': 0, 'Tb': 0, 'gear': gear_idx,
            'w': 0, 'soc': soc,
            'reward': 0, 'Tm_org': 0, 'Te_org': 0,
            'clutch': clutch, 'shift': 0, 'currency': 0,
            'cost': 0, 'p_Tm': 0, 'p_w': 0, 'p_soc': 0,
            'v': vh, 'a': ah, 'jerk': 0, 'ed': 0, 'd': dist,
            'mf': 0, 'p_vh': 0, 'p_collision': 0, 'p_ed': 0,
        }

        # np.random.seed(seed)

        return self.state, info

    def reset_zero(self, soc_init):
        """Reset the state with zero initial values"""
        # self.ep_len = len(self.ai_vec_orig)
        # self.ai_vec = np.concatenate((self.ai_vec_orig, self.ai_vec_orig), axis=0)
        # self.vi_vec = np.concatenate((self.vi_vec_orig, self.vi_vec_orig), axis=0)

        # self.state = np.array([0, 0, 0, 6, 0, soc_init, 0, 0], dtype=np.float64)
        # self.state = np.array([0, 0, 0, 6, 0, soc_init, 0, 0, 0, 0], dtype=np.float64)
        # self.state = np.array([0, 0, 0, 6, 0, soc_init, 0, 0, 0, 0, 0], dtype=np.float64)
        # self.state = np.array([0, 0, 0, 6, 0, soc_init, 0, 0, 0, 0], dtype=np.float64)
        # self.state = np.array([0, 0, 0, 6, 0, soc_init, 0, 10, self.ep_len, 0], dtype=np.float64)
        self.state = np.array([0, 0, 0, 6, 0, soc_init, 0], dtype=np.float64)
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
        # if vi == 0:  # no rolling resistance if zero velocity
        #     return wheel_radius * (m * ai
        #                            + 0.5 * Cd * rou * A * vi ** 2
        #                            + m * g * np.sin(theta))
        # else:
        #     wheel_radius * (m * ai
        #                     + 0.5 * Cd * rou * A * vi ** 2
        #                     + mu * m * g * np.cos(theta)
        #                     + m * g * np.sin(theta))
        if vi and ai == 0:
            return 0
        else:
            return wheel_radius * (m * ai
                                   + 0.5 * Cd * rou * A * vi ** 2
                                   + mu * m * g * np.cos(theta)
                                   + m * g * np.sin(theta))

    @staticmethod
    def _getacce(Tw, vi) -> float:
        """Calculate acceleration, remove rolling coefficient judgment"""
        # if vi == 0:
        #     return (
        #         Tw / wheel_radius
        #         - 0.5 * Cd * rou * A * vi ** 2
        #         - m * g * np.sin(theta)
        #     ) / m
        # else:
        #     return (
        #         Tw / wheel_radius
        #         - 0.5 * Cd * rou * A * vi ** 2
        #         - mu * m * g * np.cos(theta)
        #         - m * g * np.sin(theta)
        #     ) / m
        a = (
            Tw / wheel_radius
            - 0.5 * Cd * rou * A * vi ** 2
            - mu * m * g * np.cos(theta)
            - m * g * np.sin(theta)
        ) / m
        if vi == 0 and a < 0:
            return 0  # no backward driving
        else:
            return a

    def evaluate_on(self):
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
