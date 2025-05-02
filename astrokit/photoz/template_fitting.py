from pathlib import Path
import numpy as np

from specutils import Spectrum1D

from astrokit import datasets
from astrokit.spec import norm_spec

__all__ = [
    'save_template',
    'SWIRETemplates',
]

def save_template(path, wave, flam, wave_min=None, wave_max=None):
    if wave_min is None:
        wave_min = wave.min()
    if wave_max is None:
        wave_max = wave.max()

    if wave_min < wave.min():
        wave_extend = np.linspace(wave_min, wave.min()-1, num=100)
        flam_extend = np.zeros_like(wave_extend)
        wave = np.concatenate([wave_extend, wave])
        flam = np.concatenate([flam_extend, flam])
    elif wave_min > wave.min():
        keep_idx = wave > wave_min
        wave = wave[keep_idx]
        flam = flam[keep_idx]
    else:
        pass

    if wave_max > wave.max():
        wave_extend = np.linspace(wave.max(), wave_max, num=100)[1:]
        flam_extend = np.zeros_like(wave_extend)
        wave = np.concatenate([wave, wave_extend])
        flam = np.concatenate([flam, flam_extend])
    elif wave_max < wave.max():
        keep_idx = wave < wave_max
        wave = wave[keep_idx]
        flam = flam[keep_idx]
    else:
        pass

    data = np.column_stack((wave, flam))
    np.savetxt(path, data, fmt='%.6e', delimiter=' ')
    return path


class SWIRETemplates:
    """
    使用SWIRE模板库中的模板, 或制作新的模板

    Note
    ----
    - 可用AGN模板的名称: 
        * 'TQSO1', 'QSO1', 'BQSO1',  # Type 1 QSO (3)
        * 'QSO2', 'Torus'            # Type 2 QSO (2)
    - 可用星系模板的名称:
        * 'Ell2', 'Ell5', 'Ell13',                      # 不同年龄的椭圆星系(3)
        * 'S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Sdm', 'Spi4'   # 漩涡星系(7)
        * 'M82', 'N6090', 'N6240', 'Arp220',            # Starburst星系(4)
        * 'I22491', 'I19254', 'I20551'                  # ULIRG星系(3)
        * 'Sey18', 'Sey2', 'Mrk231'                     # Seyfert星系(3)
    """
    def __init__(self):
        self.dir_temps = Path(datasets.__path__[0]) / 'templates' / 'lephare' / 'QSO' / 'POLLETTA'
    
    def show_temp_names(self):
        ls_path_temps = list(self.dir_temps.glob('*.sed'))
        names = [path.name.split('_')[0] for path in ls_path_temps]
        return names
    
    def fetch_template(self, temp_name):
        """
        加载模板, 检查模板并归一化
        """
        path = self.dir_temps / f"{temp_name}_template_norm.sed"
        wave, flam = np.loadtxt(path, unpack=True)
        # 模板wave列重复值处理
        # 已检查: SWIRE模板库没有NAN和负流量等异常值, 也没有波长数组递减的情况; 
        # 但是有相同波长的两行数据
        unique_wave, idx = np.unique(wave, return_index=True)
        wave, flam = wave[idx], flam[idx]
        wave, flam = norm_spec(wave, flam, norm_wave=5500)

        return wave, flam

    def get_extend_UV_QSO_template(self, temp_name, 
                                   connect_point=1100, 
                                   power_law_index=0.56, 
                                   lambda_min=200):
        """
        获得蓝端延长的类星体光谱模板
        """
        wave, flam = self.fetch_template(temp_name)
        # 获取光谱中距离连接点最近的点坐标
        idx = np.abs(wave-connect_point).argmin()
        p_wave = wave[idx]
        p_flam = flam[idx]

        # 计算power-law的norm
        norm = p_flam / p_wave**(power_law_index-2)

        # 制作连接点之前的UV延长部分
        wave_extend_UV = np.arange(lambda_min, p_wave+10, 10)
        def power_law(wave, norm):
            flam = norm * (wave**(power_law_index-2))
            return flam
        flux_extend_UV = power_law(wave_extend_UV, norm)

        # 合并延长部分和原始光谱
        keep_idx = (wave > connect_point)
        keep_wave = wave[keep_idx]
        keep_flam = flam[keep_idx]
        wave = np.concatenate([wave_extend_UV, keep_wave])
        flam = np.concatenate([flux_extend_UV, keep_flam])
        return wave, flam
