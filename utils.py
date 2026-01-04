import numpy as np
import scipy.io as sio

# ALPHA = 10000
# DT = 0.1
DT = 1


CYCLE_LENGTHS = {
    'whvc': int(901 / DT),
    'chtc_lt': int(1653 / DT),
    'hd_udds': int(1061 / DT),
    'je05': int(1830 / DT),
    'hhddt': int(2084 / DT),
}

CYCLES = {
    'COWHVC': {
        'length': CYCLE_LENGTHS['whvc'],
        'ref_optimal': 0.558,  # fuel consumption
        'soc_final_ref': 0.6059,
    },
    'COCHTCLT': {
        'length': CYCLE_LENGTHS['chtc_lt'],
        'ref_optimal': 1.5031,
        'soc_final_ref': 0.5992,
        'r_ed': -73.26,
        'r_j': -54.52,
        'r_Ah': -457.42,
        # 'ref_optimal': 1.4763,
        # 'soc_final_ref': 0.5750,
    },
    'COHDUDDS': {
        'length': CYCLE_LENGTHS['hd_udds'],
        'ref_optimal': 1.123,
        'soc_final_ref': 0.6073,
    },
    'COJE05': {
        'length': CYCLE_LENGTHS['je05'],
        'ref_optimal': 1.546,
        'soc_final_ref': 0.6012,
    },
    'COHHDDT': {
        'length': CYCLE_LENGTHS['hhddt'],
        'ref_optimal': 5.242,
        'soc_final_ref': 0.6000,
    },
    'COWHVC3000': {
        'length': 3000,
        # 'dp_optimal': 0.7844 * ALPHA,
        # 'soc_final_dp': 0.5901,
    },
    'SeqEMSCHTCLT': {
        'length': CYCLE_LENGTHS['chtc_lt'],
        'ref_optimal': 1.5031,
        'soc_final_ref': 0.5992,
        'r_ed': -73.26,
        'r_j': -54.52,
        'r_Ah': -457.42,
        # 'ref_optimal': 1.4763,
        # 'soc_final_ref': 0.5750,
    },
    'SeqACCCHTCLT': {
        'length': CYCLE_LENGTHS['chtc_lt'],
        'ref_optimal': 1.5031,
        'soc_final_ref': 0.5992,
        'r_ed': -73.26,
        'r_j': -54.52,
        'r_Ah': -457.42,
        # 'ref_optimal': 1.4763,
        # 'soc_final_ref': 0.5750,
    },
    'ACCWHVC': {
        'length': 9010,
    },
    'ACCCHTCLT': {
        'length': CYCLE_LENGTHS['chtc_lt'],
    },
    'ACCWHVC3000': {
        'length': 3000,
    },
}


def least_squares(x, y, xy_lim=None):
    """linear regression by least squares method, y = a + bx"""
    n = len(x)
    sumx = 0
    sumx2 = 0
    sumy = 0
    sumy2 = 0
    sumxy = 0

    for i in range(0, n, 1):
        sumx += x[i]
        sumy += y[i]
        sumx2 += x[i] ** 2
        sumy2 += y[i] ** 2
        sumxy += x[i] * y[i]
    lxx = sumx2 - sumx ** 2 / n
    lxy = sumxy - sumx * sumy / n
    lyy = sumy2 - sumy ** 2 / n
    appro_b = lxy / lxx
    appro_a = sumy / n - appro_b * sumx / n
    R2 = lxy ** 2 / (lxx * lyy)

    x_range = max(x) - min(x)
    xnew = [min(x) - 0.2 * x_range, max(x) + 0.2 * x_range]
    result = [appro_a + appro_b * i for i in xnew]

    # # plt.style.use('seaborn-poster')
    # import matplotlib.pyplot as plt
    # plt.rcParams['figure.figsize'] = [6.4, 3.2]
    # # plt.rcParams['text.usetex'] = True
    # # plt.rcParams['font.family'] = 'serif'
    # # plt.rcParams['font.serif'] = ['Times']
    # plt.rcParams['font.size'] = 12
    # plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.scatter(x=x, y=y, color="#440154", marker='o', facecolors='none')
    # plt.plot(xnew, result, color="#22a884")
    # plt.annotate(f'y = {appro_a:.4f} + {appro_b:.4f}x',
    #              xy=(0.6, 0.3), xycoords='axes fraction')
    # # # add data point labels
    # # for xi, yi in zip(x, y):
    # #     label = f'({xi:.4f}, {yi:.4f})'
    # #     plt.annotate(label, xy=(xi, yi), xytext=(5, 5),
    # #                  textcoords='offset points', fontsize=8)
    # plt.xlabel(''r'$\Delta$ SOC')
    # plt.ylabel('Fuel consumption (kg)')
    # if isinstance(xy_lim, list):
    #     plt.xlim(xy_lim[0], xy_lim[1])
    #     plt.ylim(xy_lim[2], xy_lim[3])
    # elif xy_lim == 'scale':
    #     y_range = max(y) - min(y)
    #     xy_lim = [
    #         min(x) - 0.23 * x_range, max(x) + 0.23 * x_range,
    #         min(y) - 0.23 * y_range, max(y) + 0.23 * y_range
    #     ]
    #     plt.xlim(xy_lim[0], xy_lim[1])
    #     plt.ylim(xy_lim[2], xy_lim[3])
    # plt.tight_layout(pad=0.3)
    # # plt.savefig('fig_paper/fig-fit.png')
    # # plt.savefig('fig_paper/fig-fit.pdf')
    # plt.show()

    return appro_a, appro_b, R2, xy_lim


Ea = 31700  # activation energy, J/mol
eta = 163.3  # fitting coefficient for C-rate
R_gas = 8.314  # gas constant, J/(mol*K)
z = 0.57  # power law factor


def _get_alpha_beta(soc):
    alpha_low = 1287.6
    alpha_high = 1385.5
    beta_low = 6356.3
    beta_high = 4193.2

    alpha = np.where(soc > 0.45, alpha_high, alpha_low)
    beta = np.where(soc > 0.45, beta_high, beta_low)

    return alpha, beta


def calculate_Qloss(soc, TK, Ic, Ah):
    """Return the percentage of battery capacity loss."""
    alpha, beta = _get_alpha_beta(soc)
    Qloss = (alpha * soc + beta) * np.exp((-Ea + eta * Ic) / (R_gas * TK)) * (Ah ** z)

    return Qloss


def _calculate_batt_life(soc, TK, Ic):
    """Calculate battery life based on SOC, temperature, and c-rate
    batt_life = \Gamma=\left[\frac{20}{\left(\alpha \cdot \mathrm{SOC}+\beta\right) \cdot \exp \left(\frac{-E_a+\eta \cdot I_{c}}{R \cdot T_{\mathrm{K}}}\right)}\right]^{\frac{1}{z}}
    """
    alpha, beta = _get_alpha_beta(soc)

    return (
        (20 / ((alpha * soc + beta) * np.exp((-Ea + eta * Ic) / (R_gas * TK)))) ** (1 / z)
    )


class BattAging:
    def __init__(self):
        raise TypeError(f"'{self.__class__.__name__}' is a namespace class and cannot be instantiated.")

    T_batt = 27 + 273.15  # battery temperature, K
    BATT_LIFE_NOM = _calculate_batt_life(soc=0.6, TK=T_batt, Ic=2.5)

    @classmethod
    def calculate_severity(cls, soc, TK, Ic):
        return cls.BATT_LIFE_NOM / _calculate_batt_life(soc, TK, Ic)


def use_times_new_roman():
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    font_manager = fm.fontManager
    font_manager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()
    plt.rcParams['font.family'] = font_name


def print_results():
    # load ACC+EMS results
    mat_acc = sio.loadmat('mpc/acc_whvc_0706_1827.mat')
    mat_ems = sio.loadmat('matlab/results_hev_simplified_1whvc.mat')


if __name__ == '__main__':
    print(f'the nominal battery life is {BattAging.BATT_LIFE_NOM:.4f} Ah')
