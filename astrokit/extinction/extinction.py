"""
消光改正

@ Author: Rui Zhu
@ Date: 2025-04-09
"""
import numpy as np
import requests
import tarfile
import extinction

from astrokit.externals import sfdmap
from astrokit import DIR_data

class ExtinctionCorrection:
    def __init__(self, scaling=0.86, Rv=3.1):
        """
        消光改正模块

        Parameters
        ----------
        scaling: float
            SFD Dust Map的缩放因子, 默认值为0.86. 
            see: https://iopscience.iop.org/article/10.1088/0004-637X/725/1/1175
        
        Rv: float
            Galactic extinction coefficients, default is 3.1
        """
        self.dir_dustmap = DIR_data / 'dustmaps'
        self.dir_dustmap.mkdir(parents=True, exist_ok=True)
        self.dir_sfdmap = self.dir_dustmap / 'sfddata-master'

        if not self.dir_sfdmap.exists():
            self.download_sfdmap()

        self.scaling = scaling
        self.Rv = Rv

    def download_sfdmap(self):
        # 下载文件
        url = "https://github.com/kbarbary/sfddata/archive/master.tar.gz"

        path_sfdmaps_gz = self.dir_dustmap / "master.tar.gz"

        print(f"==> Downloading from {url}")
        response = requests.get(url)
        with open(path_sfdmaps_gz, "wb") as f:
            f.write(response.content)
        response = requests.get(url)

        # 保存下载的文件
        print(f"==> Saving to {path_sfdmaps_gz}")
        with open(path_sfdmaps_gz, "wb") as f:
            f.write(response.content)

        # 解压tar.gz文件
        print(f"==> Extracting {path_sfdmaps_gz}")
        with tarfile.open(self.dir_dustmap / "master.tar.gz", "r:gz") as tar:
            tar.extractall(path=self.dir_dustmap, filter="tar")  # 解压到sfddata文件夹
        
        # 删除tar.gz文件
        print(f"==> Removing {path_sfdmaps_gz}")
        path_sfdmaps_gz.unlink()
        print(f"==> SFD maps downloaded!")

        return None

    def cal_ebv(self, ra, dec):
        sfd = sfdmap.SFDMap(mapdir=self.dir_sfdmap, scaling=self.scaling)
        ebv = sfd.ebv(ra, dec, frame='icrs', unit='degree')
        return ebv
    
    def cal_extinction(self, ra, dec, filter_waves):
        """
        基于SFD Dust Maps和Fitzpatrick & Massa (2007)的消光曲线

        Parameters  
        ----------
        ra: np.ndarray
            ra list in ICRS, unit: degree
        dec: np.ndarray
            dec list in ICRS, unit: degree
        filter_waves: list
            filter central wavelengths, unit: Angstrom
        """
        # 计算V-band extinction
        Av = self.cal_ebv(ra, dec) * self.Rv

        # 从extinction模块中的的extinction cure得到A_x = Av * coeff
        coeff = extinction.fm07(np.array(filter_waves), 1.0)  # fm07固定Rv=3.1
        Ax = Av.reshape(-1, 1) * coeff

        output = []
        for i in range(len(filter_waves)):
            output.append(Ax[:, i])
        return output