"""
EAZY Wrapper

@ Author: Rui Zhu
"""
import re
import time
from pathlib import Path
from loguru import logger
import subprocess
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 50

from astropy.table import Table

from astrokit import CONFIG
from astrokit.toolbox import value_to_KVD_string
from astrokit.toolbox import run_cmd_in_terminal
from astrokit.toolbox import fnu_to_flam, flam_to_fnu, fnuErr_to_flamErr

from astrokit.phot import EazyFilters

__all__ = [
    'EazyOutput', 
    'Eazy'
]


class EazyOutput:
    """
    集中管理EAZY的输出文件
    
    Note: 
    - 代码参考: /Users/rui/Applications/eazy-photoz/inputs/read_eazy_binaries.py
    """
    def __init__(self, dir_output):
        self.dir_output = Path(dir_output)

        self.tempfilt = self.fetch_tempfilt()
        self.coeff = self.fetch_coeff()
        self.temp_sed = self.fetch_temp_sed()
        self.pz = self.fetch_pz()
        self.zbin = self.fetch_zbin()

        self.EAZY_res = self.collect()
    
    def fetch_zout(self, fname='photz.zout'):
        tbl = Table.read(self.dir_output / fname, format='ascii.commented_header')
        df = tbl.to_pandas()
        df.loc[df['z_spec'] < 0, 'z_spec'] = np.NaN
        # df.loc[df['z_a'] < 0, 'z_a'] = np.NaN
        return df
    
    def fetch_tempfilt(self, fname="photz.tempfilt"):
        with open(self.dir_output / fname, 'rb') as f:
            s = np.fromfile(file=f, dtype=np.int32, count=4)
            NFILT = s[0]  # number of filters
            NTEMP = s[1]  # number of templates
            NZ = s[2]  # number of redshifts
            NOBJ = s[3]  # number of objects
            tempfilt = np.fromfile(file=f, dtype=np.double, count=NFILT*NTEMP*NZ).reshape((NZ, NTEMP, NFILT))
            lc = np.fromfile(file=f, dtype=np.double, count=NFILT)
            zgrid = np.fromfile(file=f, dtype=np.double, count=NZ)
            fnu = np.fromfile(file=f, dtype=np.double, count=NFILT*NOBJ).reshape((NOBJ, NFILT))
            efnu = np.fromfile(file=f, dtype=np.double, count=NFILT*NOBJ).reshape((NOBJ, NFILT))
        self.EAZY_res = {
            'NFILT': NFILT,
            'NTEMP': NTEMP,
            'NZ': NZ,
            'NOBJ': NOBJ, 
            'tempfilt': tempfilt,  # fluxes derievd from templates at different bandpass
            'lc': lc,  # mean wavelength of each filters
            'zgrid': zgrid,  # redshift grid
            'fnu': fnu,  # 每个源的观测F_nu, 单位是erg s-1 cm-1 Hz-1, 即使输入的是mag，也会转换成f_nu
            'efnu': efnu  # flux error of galaxies
            }
        return self.EAZY_res
    
    def fetch_coeff(self, fname="photz.coeff"):
        with open(self.dir_output / fname, 'rb') as f:
            s = np.fromfile(file=f, dtype=np.int32, count=4)
            NFILT = s[0]
            NTEMP = s[1]
            NZ = s[2]
            NOBJ = s[3]
            coeff = np.fromfile(file=f, dtype=np.double, count=NTEMP*NOBJ).reshape((NOBJ, NTEMP))
            izbest = np.fromfile(file=f, dtype=np.int32, count=NOBJ)
            tnorm = np.fromfile(file=f, dtype=np.double, count=NTEMP)

        self.EAZY_res = {
            'NFILT': NFILT, 
            'NTEMP': NTEMP,
            'NZ': NZ,
            'NOBJ': NOBJ, 
            'coeff': coeff,  # coefficients of each template for each object
            'izbest': izbest,  # the index of the best redshift in zgrid
            'tnorm': tnorm  # 在threedhst-eazyPy中被用于估计stellar mass
            }
        return self.EAZY_res
    
    def fetch_temp_sed(self, fname="photz.temp_sed"):
        with open(self.dir_output / fname, 'rb') as f:
            s = np.fromfile(file=f, dtype=np.int32, count=3)
            NTEMP = s[0]
            NTEMPL = s[1]
            NZ = s[2]
            temp_wave = np.fromfile(file=f, dtype=np.double, count=NTEMPL)
            temp_flam = np.fromfile(file=f, dtype=np.double, count=NTEMPL*NTEMP).reshape((NTEMP,NTEMPL))
            da = np.fromfile(file=f, dtype=np.double, count=NZ)
            db = np.fromfile(file=f, dtype=np.double, count=NZ)

        self.EAZY_res = {
            'NTEMP': NTEMP, 
            'NTEMPL': NTEMPL,  # length of template wavelength array
            'NZ': NZ,
            'temp_wave': temp_wave,  # templates wavelength array
            'temp_flam': temp_flam,  # flambda of all the templates (经过差值处理后)
            'da': da,  # Da
            'db': db  # Db
            }
        return self.EAZY_res
    
    def fetch_pz(self, fname='photz.pz'):
        pz = dict()
        with open(self.dir_output / fname, 'rb') as f:
            NZ, NOBJ = np.fromfile(file=f, dtype=np.int32, count=2)
            chi2fit = np.fromfile(file=f, 
                                dtype=np.double, 
                                count=NZ*NOBJ).reshape((NOBJ, NZ))
            pz['NZ'] = NZ
            pz['NOBJ'] = NOBJ
            pz['chi2fit'] = chi2fit

            s = np.fromfile(file=f, dtype=np.int32, count=1)

            if len(s) == 0:  # no prior case
                pz['NK'] = None
                pz['kbins'] = None
                pz['priorzk'] = None  # 行: 红移采样; 列: mag bin; 指定红移, mag bin处的先验值(即先验文件对应的数组)
                pz['kidx'] = None
                pz['priorz'] = np.ones([NOBJ, NZ])
            else:
                NK = s[0]  # number of prior's mag bins
                pz['NK'] = NK
                pz['kbins'] = np.fromfile(file=f, dtype=np.double, count=NK)
                pz['priorzk'] = np.fromfile(file=f, 
                                            dtype=np.double, 
                                            count=NZ*NK).reshape((NK,NZ)).transpose()
                pz['kidx'] = np.fromfile(file=f, dtype=np.int32, count=NOBJ)
                # 每个源的红移先验
                kidx = pz['kidx'] # 每个源对应的先验的mag bin的索引
                err_idx = (kidx >= NK) | (kidx < -1)  # 先验的mag bin的索引是否超出了mag bin的数量
                kidx = np.where(err_idx, 0, kidx)  # 先将有问题的索引成0，不然索引不到
                priorz = pz['priorzk'][:, kidx].T  # 每个源对应的红移先验概率
                priorz[err_idx] = np.ones(NZ)  # 超出mag bin的索引的源, 先验概率设置为1
                pz['priorz'] = priorz

        return pz
    
    def fetch_zbin(self, fname='photz.zbin'):
        with open(self.dir_output / fname, 'rb') as f:
            s = np.fromfile(file=f, dtype=np.int32, count=1)
            NOBJ = s[0]
            z_a = np.fromfile(file=f, dtype=np.double, count=NOBJ)
            z_p = np.fromfile(file=f, dtype=np.double, count=NOBJ)
            z_m1 = np.fromfile(file=f, dtype=np.double, count=NOBJ)
            z_m2 = np.fromfile(file=f, dtype=np.double, count=NOBJ)
            z_peak = np.fromfile(file=f, dtype=np.double, count=NOBJ)
        self.EAZY_res = {
            'NOBJ': NOBJ,  # number of objects
            'z_a': z_a,  # average z
            'z_p': z_p,  # median z
            'z_m1': z_m1,  # z_low
            'z_m2': z_m2,  # z_high
            'z_peak': z_peak  # z_peak
            }
        return self.EAZY_res

    def collect(self):
        """
        - NOBJ: number of objects
        - NFILT: number of filters
        - NTEMP: number of templates
        - NZ: number of redshifts (zgrid)

        ## From tempfilt file
        - lc: mean wavelength of each filters
        - zgrid: redshift grid
        - tempfilt: fluxes derievd from templates at different bandpass
        - fnu: 每个源的观测F_nu, 单位是erg s-1 cm-1 Hz-1, 即使输入的是mag, 也会转换成f_nu
        - efnu: flux error of galaxies

        ## From coeff file
        - coeff: coefficients of each template for each object
        - izbest: the index of the best redshift in zgrid
        - tnorm: 在threedhst-eazyPy中被用于估计stellar mass

        ## From temp_sed file
        - temp_wave: 输入模板的wave
        - temp_flam: 输入模板的flam
        - da: Da
        - db: Db

        ## From pz file
        - NK: number of prior's mag bins
        - chi2fit: chi2 of each redshift for each object
        - kbins: prior's mag bins
        - priorzk: prior's redshift distribution
        - kidx: prior's mag bin index for each object

        ## From zbin file
        - z_a: average z
        - z_p: median z
        - z_m1: z_low
        - z_m2: z_high
        - z_peak: z_peak
        """
        self.EAZY_res = {
            'NOBJ': self.tempfilt['NOBJ'],    # number of objects
            'NFILT': self.tempfilt['NFILT'],  # number of filters
            'NTEMP': self.tempfilt['NTEMP'],  # number of templates
            'NZ': self.tempfilt['NZ'],        # number of redshifts (zgrid)
            
            # from tempfilt file
            'lc': self.tempfilt['lc'],               # mean wavelength of each filters
            'zgrid': self.tempfilt['zgrid'],         # redshift grid
            'tempfilt': self.tempfilt['tempfilt'],   # fluxes derievd from templates at different bandpass
            'fnu': self.tempfilt['fnu'],             # 每个源的观测F_nu, 单位是erg s-1 cm-1 Hz-1, 即使输入的是mag，也会转换成f_nu
            'efnu': self.tempfilt['efnu'],           # flux error of galaxies
            
            # from coeff file
            'coeff': self.coeff['coeff'],            # coefficients of each template for each object
            'izbest': self.coeff['izbest'],          # the index of the best redshift in zgrid
            'tnorm': self.coeff['tnorm'],            # 在threedhst-eazyPy中被用于估计stellar mass
            
            # from temp_sed file
            'temp_wave': self.temp_sed['temp_wave'],  # templates wavelength array
            'temp_flam': self.temp_sed['temp_flam'],  # flambda of all the templates (经过差值处理后)
            'da': self.temp_sed['da'],                # Da
            'db': self.temp_sed['db'],                # Db

            # from pz file
            'NK': self.pz['NK'],                     # number of prior's mag bins
            'chi2fit': self.pz['chi2fit'],           # chi2 of each redshift for each object
            'kbins': self.pz['kbins'],               # prior's mag bins
            'priorzk': self.pz['priorzk'],           # prior's redshift distribution
            'kidx': self.pz['kidx'],                 # prior's mag bin index for each object
            'priorz': self.pz['priorz'],             # redshift prior for each object

            # from zbin file
            'z_a': self.zbin['z_a'],         # average z
            'z_p': self.zbin['z_p'],         # median z
            'z_m1': self.zbin['z_m1'],       # z_low
            'z_m2': self.zbin['z_m2'],       # z_high
            'z_peak': self.zbin['z_peak']    # z_peak
        }
        return self.EAZY_res

    def get_obj_res(self, idx):
        """
        获取索引为idx的目标源的EAZY拟合结果

        Parameters
        ----------
        idx: int
            目标源在catalog中的index, 从0开始
        apply_IGM_correction: bool
            是否对模板进行IGM修正

        Return
        ------
        1. NFILT: 使用的filter的数量
        2. NTEMP: 使用的模板的数量
        3. NZ: 红移采样数量
        4. lc: filter的中心波长
        5. coeff: 每个模板的最佳拟合系数
        6. obs_fnu: catalog里的测光值转换成f_nu, 单位为erg s-1 cm-2 Hz-1
        7. obs_efnu: catalog里的测光值转换成f_nu的误差, 单位为erg s-1 cm-2 Hz-1
        8. obs_flambda: catalog里的测光值转换成f_lambda, 单位为erg s-1 cm-2 AA-1
        9. obs_eflambda: catalog里的测光值转换成f_lambda的误差, 单位为erg s-1 cm-2 AA-1
        10. sim_fnu: 模板计算得到的模拟的测光点, 单位为erg s-1 cm-2 Hz-1
        11. sim_flambda: 模板计算得到的模拟的测光点, 单位为erg s-1 cm-2 AA-1
        12. z_best: 最佳红移
        13. lambdaz: 移动模板到观测的波长
        14. sim_sed_fnu: 组合多个模板生成的最佳拟合的SED, 单位为erg s-1 cm-2 Hz-1
        15. sim_sed_flambda: 组合多个模板生成的最佳拟合的SED, 单位为erg s-1 cm-2 AA-1

        """
        # from tempfilt
        tempfilt = self.EAZY_res['tempfilt']
        fnu = self.EAZY_res['fnu']
        fnu[fnu<0] = np.nan
        efnu = self.EAZY_res['efnu']
        efnu[efnu<0] = np.nan
        lc = self.EAZY_res['lc']
        zgrid = self.EAZY_res['zgrid']

        # from temp_sed
        temp_wave = self.EAZY_res['temp_wave']  # 模板的波长, 静止系
        temp_flam = self.EAZY_res['temp_flam']  # 全部模版的SED, 单位为flambda
        da = self.EAZY_res['da']
        db = self.EAZY_res['db']

        # from coeff
        izbest = self.EAZY_res['izbest']
        coeff = self.EAZY_res['coeff']

        # ----- 检索目标源信息 -----
        izbest = izbest[idx]
        z_best = zgrid[izbest]  # 最佳红移
        lambdaz = (1+z_best) * temp_wave  # 移动模板到观测的波长
        tempfilt = tempfilt[izbest].T
        coeff = coeff[idx]

        # 模板计算得到的模拟的测光点
        sim_fnu = np.dot(tempfilt, coeff)
        sim_flam = fnu_to_flam(fnu=sim_fnu, lambda_c=lc)

        # 观测的测光点(from catalog)
        obs_fnu = fnu[idx]
        obs_efnu = efnu[idx]
        obs_flam = fnu_to_flam(fnu=obs_fnu, lambda_c=lc)
        obs_eflam = fnuErr_to_flamErr(fnu_err=obs_efnu, lambda_c=lc)

        # IGM correction for SED
        def apply_IGM_correction(flam):
            lim1 = np.where(temp_wave < 912)[0]
            lim2 = np.where((temp_wave >= 912) & (temp_wave < 1026))[0]
            lim3 = np.where((temp_wave >= 1026) & (temp_wave < 1216))[0]
            if lim1.size > 0:
                flam[lim1] *= 0
            if lim2.size > 0:
                flam[lim2] *= 1 - db[izbest]
            if lim3.size > 0:
                flam[lim3] *= 1 - da[izbest]
            return flam

        # 组合多个模板生成的最佳拟合的SED
        c = 3E18  # speed of light in Angstrom/sec
        sim_sed_flam_component = []
        sim_sed_fnu_component = []
        for i in range(len(temp_flam)):
            flam_comp = temp_flam[i] * coeff[i]
            flam_comp = flam_comp / (1+z_best)**2  # 考虑flux随红移衰减
            flam_comp = flam_comp * (1/5500)**2 * c
            flam_comp = apply_IGM_correction(flam_comp)
            sim_sed_flam_component.append(flam_comp)
            sim_sed_fnu_component.append(flam_to_fnu(flam=flam_comp, lambda_c=lambdaz))
        sim_sed_flam = np.sum(sim_sed_flam_component, axis=0)
        sim_sed_fnu = flam_to_fnu(flam=sim_sed_flam, lambda_c=lambdaz)
            
        # ----- 收集结果 -----
        res = dict()  # 结果收集器
        res['NFILT'] = self.EAZY_res['NFILT']
        res['NTEMP'] = self.EAZY_res['NTEMP']
        res['NZ'] = self.EAZY_res['NZ']
        res['lc'] = lc
        res['coeff'] = coeff  # 最佳拟合的系数

        res['obs_fnu'] = obs_fnu
        res['obs_efnu'] = obs_efnu
        res['obs_flam'] = obs_flam
        res['obs_eflam'] = obs_eflam

        res['sim_fnu'] = sim_fnu
        res['sim_flam'] = sim_flam

        res['z_best'] = z_best
        res['lambdaz'] = lambdaz
        res['sim_sed_fnu'] = sim_sed_fnu
        res['sim_sed_flam'] = sim_sed_flam
        res['sim_sed_fnu_component'] = sim_sed_fnu_component
        res['sim_sed_flam_component'] = sim_sed_flam_component
        return res
    
    def get_obj_zPDF(self, idx):
        # 将红移随chi2fit的分布, 转换成红移的概率分布
        zgrid = self.EAZY_res['zgrid']
        chi2fit = self.EAZY_res['chi2fit'][idx]
        priorz = self.EAZY_res['priorz'][idx]

        min_chi2fit = np.min(chi2fit)
        pz_orig = np.exp(-0.5 * (chi2fit - min_chi2fit))
        pz_orig = pz_orig / np.trapz(pz_orig, zgrid)

        # 添加先验后的计算可能有问题, 有需求再改
        pz_prior = pz_orig * priorz
        pz_prior = pz_prior / np.trapz(pz_prior, zgrid)

        zPDF = pd.DataFrame({
            'z': zgrid, 
            'chi2': chi2fit, 
            'pz_orig': pz_orig, 
            'pz_prior': pz_prior
        })
        return zPDF
    
    def plot_SED(self, idx, temp_names=None, ax=None, return_ax=False):
        obj_res = self.get_obj_res(idx)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        # 确定x范围
        XTICKS = [
            100, 500, 800, 
            1000, 2000, 3000, 4000, 5000, 7000, 
            10000, 20000, 30000, 40000, 60000, 100000, 
            200000, 300000, 500000, 700000, 1000000]
        xlim = [max([i for i in XTICKS if i < obj_res['lc'].min()]),
                min([i for i in XTICKS if i > obj_res['lc'].max()])]
        ax.set_xlim(xlim)
        ax.set_xscale('log')
        xticks = [i for i in XTICKS if i >= xlim[0] and i <= xlim[1]]
        ax.set_xticks(ticks=xticks, labels=xticks)

        # 实际观测的测光点
        ax.errorbar(obj_res['lc'], obj_res['obs_flam'], yerr=obj_res['obs_eflam'], 
                    fmt='o', color='k', ms=4, linewidth=2, capsize=6, zorder=1, label="Observed Flux")

        # EAZY SED
        # 选择要展示的数据
        idx_sed = np.where((obj_res['lambdaz'] >= xlim[0]) & (obj_res['lambdaz'] <= xlim[1]))[0]
        ax.plot(obj_res['lambdaz'][idx_sed], obj_res['sim_sed_flam'][idx_sed], lw=1.5, alpha=1, 
                zorder=-1, c='green', label="Best-fit SED")
        # 绘制子成分
        colors = ['blue', 'red', 'purple', 'orange', 'brown']
        for i in range(len(obj_res['sim_sed_flam_component'])):
            if temp_names is None:
                label = None
            else:
                label = f"{temp_names[i]}"
            ax.plot(obj_res['lambdaz'][idx_sed], obj_res['sim_sed_flam_component'][i][idx_sed], 
                    lw=0.5, alpha=1, zorder=0, c=colors[i], ls='-', label=label)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(axis='x', top=True, which='major', width=1.5, length=5, direction='in')
        ax.tick_params(axis='x', top=True, which='minor', width=1.5, length=3, direction='in')
        ax.tick_params(axis='y', which='major', width=1.5, length=5, direction='in')
        ax.tick_params(axis='y', which='minor', width=1.5, length=3, direction='in')
        ax.set_xlabel(r"Observed Wavelength ($\rm \AA$)", fontsize=15)
        ax.set_ylabel(r"$\rm f_{\lambda} (erg s^{-1} cm^{-2} \AA^{-1})$", fontsize=15)
        ax.legend(fontsize=12, loc='upper right', frameon=False)

        if return_ax:
            return ax
        else:
            None


class Eazy:
    """A Wrapper for EAZY"""
    def __init__(
            self, 
            dir_work, 
            catalog,
            translate_dict, 
            fname_catalog: str = "catalog.cat",
            fname_translate: str = "photz.translate", 
            fname_config: str =  "photz.param", 
            fname_spectra_param = 'templates.spectra.param', 
            fname_output_zout: str = "photz.zout", 
            silence=False
            ):
        """
        Parameters
        ----------
        dir_work:
            EAZY任务的执行目录
        catalog:
            输入星表, 异常值或缺失值标记为np.NaN
            注意, 星表必须包含以下几个列名:
            1. id: 每个源的识别码
            2. z_spec: 光谱红移
            3. mag_: AB星等
            4. magerr_: AB星等的误差
            这里统一使用AB星等, 避免因zeropoint设置值错误造成难以发现的异常; 
            同时提醒运行前进行数据检查
        translate_dict:
            使用的filter名称与其在EAZY filter数据文件中的id的对应字典
        """
        self.silence = silence

        # * 创建目录结构
        self.dir_work = Path(dir_work)
        self.dir_output = self.dir_work / 'output'

        for path in [self.dir_work, self.dir_output]:
            if not path.exists():
                path.mkdir(exist_ok=True)

        self.fname_catalog = fname_catalog
        self.fname_translate = fname_translate
        self.fname_config = fname_config
        self.fname_spectra_param = fname_spectra_param
        self.fname_output_zout = fname_output_zout

        # input
        self.EazyFilters = EazyFilters(translate_dict)
        self.used_bands = self.EazyFilters.info()['filter_name'].tolist()
        self.set_bands()

        self.catalog_orig = catalog.copy()
        self.catalog = self._fetch_input_catalog()

    def _fetch_input_catalog(self):
        """
        输入星表检查, 并在工作目录下写入EAZY格式的星表文件
        """
        df = self.catalog_orig.copy()

        if 'id' not in df.columns:
            raise ValueError("id column not found!")
        if 'z_spec' not in df.columns:
            raise ValueError('z_spec column not found!')

        need_cols = ['id', 'z_spec']
        for band in self.used_bands:
            need_cols.append(f"mag_{band}")
            need_cols.append(f"magerr_{band}")
        df = df[need_cols]

        tbl = Table.from_pandas(df.replace(np.NaN, -99))
        tbl.write(self.dir_work / self.fname_catalog, 
                  format='ascii.commented_header', overwrite=True)
        return df
    
    def get_output(self):
        if len(list(self.dir_output.iterdir())) == 0:
            output = None
        else:
            output = EazyOutput(self.dir_output)
        return output
    
    # **********
    # 处理配置文件
    # **********

    def default_config_file(self, show=True):
        """
        Show the default config file of EAZY.
        """
        path = Path(CONFIG['PATH_EAZY']) / 'inputs' / 'zphot.param.default'
        if not path.exists():
            raise FileNotFoundError(f"{path}")
        else:
            with open(path, "r") as f:
                config_file = f.read()
        if show:
            print(config_file)
            return None
        else:
            return config_file
    
    def _default_config(self):
        """
        从配置文件模板中提取默认的配置信息,
        """
        config_params = dict()
        config_params_class = dict()
        
        content = re.split(
            pattern=r"\n+##\s.+\n", 
            string=self.default_config_file(show=False)
            )
        param_class = re.findall(
            pattern=r"##\s.+", 
            string=self.default_config_file(show=False)
            )
        i = 1
        for string_group in content[1:]:
            config_params_class[param_class[i].strip('## ')] = list()
            content_group = re.split(pattern=r"\n?\n(?=\w)", 
                                     string=string_group)
            for content_item in content_group:
                key, value, comment = re.split(pattern=r"\s+", 
                                               string=content_item, 
                                               maxsplit=2)
                config_params[key] = dict(value=value, comment=comment)
                config_params_class[param_class[i].strip('## ')].append(key)
            i += 1

        # 将所有默认文件都设置成绝对路径
        path_temps = Path(CONFIG['PATH_EAZY']) / 'templates'
        path_filters = Path(CONFIG['PATH_EAZY']) / 'filters'
        path_demo = Path(CONFIG['PATH_EAZY']) / 'inputs'

        config_params['FILTERS_RES']['value'] = str(path_filters / 'FILTER.RES.latest')
        config_params['TEMPLATES_FILE']['value'] = str(path_temps / 'eazy_v1.2_dusty.spectra.param')
        config_params['WAVELENGTH_FILE']['value'] = str(path_temps / 'EAZY_v1.1_lines' / 'lambda_v1.1.def')
        config_params['TEMP_ERR_FILE']['value'] = str(path_temps / 'TEMPLATE_ERROR.eazy_v1.0')
        config_params['LAF_FILE']['value'] = str(path_temps / 'LAFcoeff.txt')
        config_params['DLA_FILE']['value'] = str(path_temps / 'DLAcoeff.txt')
        config_params['CATALOG_FILE']['value'] = str(path_demo / 'hdfn_fs99_eazy.cat')
        config_params['PRIOR_FILE']['value'] = str(path_temps / 'prior_K_extend.dat')

        return config_params, config_params_class

    def config(self, **kwargs):
        """
        填写`zphot.param`配置文件
        """
        # 获取默认的配置参数字典
        config_params, config_params_class = self._default_config()

        # 修改必填项
        config_params['MAGNITUDES']['value'] = 'y'
        config_params['PRIOR_ABZP']['value'] = '-48.6'
        config_params['CATALOG_FILE']['value'] = str(self.dir_work / self.fname_catalog)
        config_params['TEMPLATES_FILE']['value'] = str(self.dir_work / self.fname_spectra_param)
        config_params['OUTPUT_DIRECTORY']['value'] = str(self.dir_output)
        config_params['MAIN_OUTPUT_FILE']['value'] = Path(self.fname_output_zout).stem

        # 修改补充修改项
        modified_params = dict()
        for key, value in kwargs.items():
            if key in ['MAGNITUDES',
                       'PRIOR_ABZP',
                       'CATALOG_FILE', 
                       'TEMPLATES_FILE', 
                       'OUTPUT_DIRECTORY', 
                       'MAIN_OUTPUT_FILE', 
                       ]:
                raise KeyError(f"No need for repetitive input of the {key} parameter!")
            if key not in config_params.keys():
                raise ValueError(f"The keyword '{key}' is not valid for EAZY!")
            else:
                old_value = config_params[key]['value']
                config_params[key]['value'] = value_to_KVD_string(value)
                modified_params[key] = (config_params[key]['value'], old_value)

        # 回传重要变量
        self.config_params = config_params
        self.TEMPLATE_COMBOS = config_params['TEMPLATE_COMBOS']['value']

        # 生成配置文件
        # edit the configuration file
        content = "# Configuration File for EAZY\n"
        content += "# Build by AstroKit\n"
        
        config_params['TEMPLATE_COMBOS']['comment'] = re.sub(
            pattern=r"\s#", repl=" "*15 + "#", 
            string=config_params['TEMPLATE_COMBOS']['comment']
        )
        for param_class in config_params_class.keys():
            content += f"\n#{param_class:^80}\n".replace(' ', '-')
            for param in config_params_class[param_class]:
                content += f"{param:<20}  {config_params[param]['value']:<30}  {config_params[param]['comment']}\n"
        
        # write the configuration file
        path = self.dir_work / self.fname_config
        with open(path, 'w') as f:
            f.write(content)
        
        if not self.silence:
            logger.info(f"Config File: {path}")

        return None
    
    # **********
    # 处理template
    # **********
    def default_EAZY_templates(self):
        """
        load the default templates of EAZY from the EAZY package
        `EAZY_v1.1_lines`
        """
        path = Path(CONFIG['PATH_EAZY']) / 'templates' / 'EAZY_v1.1_lines'
        return sorted(list(path.glob('*.dat')))

    def set_templates(
            self, 
            templates: list,  
            lambda_factor=1.0, 
            model_age=0, 
            template_error_amplitude=1.0,
            comb_dict=None
            ):
        """
        Set templates for EAZY and create the template definition file `xxx.spectra.param`.

        Parameters
        ----------
        templates : list
            A list of template path.
        lambda_factor : float, optional
            Multiplicative factor to correct wavelength units, by default 1.0
        model_age : float, optional
            Age of template model in Gyr (0 means template is always used), by default 0
        template_error_amplitude : float, optional
            Template error amplitude (for INDIVIDUAL template fits), by default 1.0
        comb_dict : dict, optional
            手动设置模板组合的方式(从1开始计数, 与tempaltes列表顺序一致), by default None; 
            设置为None, 则自动填写为全排列;
            格式为{1: [2, 3, 4], 2: [3, 4], 3: [4]}, 表示1号模板与2, 3, 4号模板组合, 
            2号模板与3, 4号模板组合, 3号模板与4号模板组合       
        """
        # check
        for path in templates:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Template file {path} not found.")
            
        # edit the template definition file
        content = []
        content.append("# Template definition file")
        content.append("# Build by AstroKit")
        content.append("#")
        content.append("# Column definitions:")
        content.append("#   1. Template number")
        content.append("#   2. Template file name")
        content.append("#   3. Lambda_conv (multiplicative factor to correct wavelength units)")
        content.append("#   4. Age of template model in Gyr (0 means template is always used)")
        content.append("#   5. Template error amplitude (for INDIVIDUAL template fits)")
        content.append("#   6. Comma/space separated list of template numbers to be combined")
        content.append("#      with current template if combined fits are enabled.")
        content.append("#")
        max_length = max([len(str(path)) for path in templates])
        for i in range(len(templates)):
            row_str = f"{i+1:<3}   "
            row_str += f"{str(templates[i]):<{max_length}}   "
            row_str += f"{lambda_factor} "
            row_str += f"{model_age} "
            row_str += f"{template_error_amplitude}   "
            combined_num = []
            if comb_dict is None:
                j = i+1
                while j < len(templates):
                    combined_num.append(j+1)
                    j += 1
            else:
                if i+1 in comb_dict.keys():
                    combined_num = comb_dict[i+1]
                else:
                    combined_num = []
            combined_num = str(combined_num).strip("[]").replace(" ", "")
            row_str += f" {combined_num}"
            content.append(row_str)

        path = self.dir_work / self.fname_spectra_param
        with open(path, "w") as f:
            f.write("\n".join(content))
            if not self.silence:
                logger.info(f"Template Definition File: {path}")
        self.templates = templates

        return None

    # **********
    # 处理filter
    # **********
    def set_bands(self):
        content = str()
        for idx, row in self.EazyFilters.info().iterrows():
            content += f"{'mag_'+row['filter_name']:<15} F{row['filter_idx']}\n"
            content += f"{'magerr_'+row['filter_name']:<15} E{row['filter_idx']}\n"

        path = self.dir_work / self.fname_translate
        with open(path, 'w') as f:
            f.write(content)
            if not self.silence:
                logger.info(f"Filter Translate File: {path}")

        return None
    
    # ********
    # 运行
    # ********
    def run(self, terminal=True):

        cmd = f"cd {self.dir_work}\n"
        cmd += f"eazy -p {self.fname_config} -t {self.fname_translate}"
        st = time.time()
        if terminal:
            if not self.silence:
                logger.info("Launch EAZY to the terminal.")
            run_cmd_in_terminal(cmd)
        else:
            if not self.silence:
                logger.info("running EAZY...")
            res = subprocess.run(args=cmd, shell=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                universal_newlines=True)
            if not self.silence:
                if "Done fitting redshifts." in res.stdout:
                    logger.success(f"EAZY finished! Cost Time: {time.time()-st:.2f} s")
                else:
                    logger.error(f"EAZY failed! {res.stderr}")
