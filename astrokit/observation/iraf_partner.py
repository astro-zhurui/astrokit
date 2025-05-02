from pathlib import Path
import shutil
from astropy.io import fits

import pytz
from astropy.time import Time

import pandas as pd

__all__ = [
    'IRAFPartner',
]

class IRAFPartner:
    """
    A Helper for IRAF to process the spectra

    Parameters
    ----------
    dir_orig : str
        The directory where the original data is stored
    dir_process : str
        The directory where the processed data will be stored
    """
    def __init__(self, dir_orig, dir_process):
        self.dir_orig = Path(dir_orig)
        self.dir_process = Path(dir_process)

        # 全部文件名
        orig_files = [path.name for path in list(self.dir_orig.glob("*.fit"))]
        orig_files.sort()
        self.orig_files = orig_files

        self.file_dict = {
            'bias': [],
            'flat': [],
            'lamp': [],
            'std_star': [],
            'target': []
        }
    
    def _check_file_exist(self, fname):
        if not (self.dir_orig / fname).exists():
            raise FileNotFoundError(f"{fname} does not exist in {self.dir_orig}")
    
    def ut2bj(self, ut_time):
        """
        Convert UT time to Beijing time
        """
        ut_time = Time(ut_time, format='isot', scale='utc')
        ut_time = ut_time.to_datetime()
        ut_time = pytz.utc.localize(ut_time)
        bj_time = ut_time.astimezone(pytz.timezone('Asia/Shanghai'))
        return bj_time.strftime('%Y-%m-%d %H:%M:%S')
    
    def find_file(self, name, file_type):
        for file in self.orig_files:
            if (name in file) and (file_type in file):
                return file
        return None
    
    def set_bias(self, fname_list):
        for i, fname in enumerate(fname_list):
            self._check_file_exist(fname)
            fname_new = f"bias_{i}.fits"
            self.file_dict["bias"].append(fname_new)
            shutil.copy(self.dir_orig / fname, self.dir_process / fname_new)
    
    def set_flat(self, fname_list):
        for i, fname in enumerate(fname_list):
            self._check_file_exist(fname)
            fname_new = f"flat_{i}.fits"
            self.file_dict["flat"].append(fname_new)
            shutil.copy(self.dir_orig / fname, self.dir_process / fname_new)
    
    def set_lamp(self, fname_list):
        for i, fname in enumerate(fname_list):
            self._check_file_exist(fname)
            fname_new = f"lamp_{i}.fits"
            self.file_dict["lamp"].append(fname_new)
            shutil.copy(self.dir_orig / fname, self.dir_process / fname_new)

    def set_standard_star(self, name_list, file_type="SPECLFLUXREF"):
        for name in name_list:
            fname = None
            for file in self.orig_files:
                if (name in file) and (file_type in file):
                    fname = file
            if fname is None:
                raise FileNotFoundError(f"Cannot find the standard star for {name}")
            self._check_file_exist(fname)
            fname_new = f"std_star_{name}.fits"
            self.file_dict["std_star"].append(fname_new)
            shutil.copy(self.dir_orig / fname, self.dir_process / fname_new)
            
    def set_target(self, name, fname_list: list):
        if len(fname_list) == 1:
            fname_orig = fname_list[0]
            fname_new = f"target_{name}.fits"
            self._check_file_exist(fname_orig)
            self.file_dict["target"].append(fname_new)
            shutil.copy(self.dir_orig / fname_orig, self.dir_process / fname_new)
        else:
            for i, fname in enumerate(fname_list):
                self._check_file_exist(fname)
                fname_new = f"target_{name}_p{i}.fits"
                self.file_dict["target"].append(fname_new)
                shutil.copy(self.dir_orig / fname, self.dir_process / fname_new)

    def header_info(self):
        data = []
        params = [
            'OBJECT', 'NAXIS1', 'NAXIS2', 'DATE-OBS', 
            'EXPTIME', 'GAIN', 'RDNOISE', 'AIRMASS', 'FILTER'
        ]
        for file_class in ['target', 'std_star', 'bias', 'flat', 'lamp']:
            for fname in self.file_dict[file_class]:
                item = {}
                item["file_class"] = file_class
                item["fname"] = fname
                with fits.open(self.dir_process / fname) as hdul:
                    header = hdul[0].header
                    for param in params:
                        item[param] = header.get(param, '--')
                
                data.append(item)

        df = pd.DataFrame(data)
        df['OBS_TIME'] = df['DATE-OBS'].apply(self.ut2bj)
        df = df.drop(columns=['DATE-OBS'])
        df.insert(3, 'OBS_TIME', df.pop('OBS_TIME'))
        return df
    
    def make_list(self):
        # list for 合并本底
        files = self.file_dict['bias']
        with open(self.dir_process / 'list_bias', 'w') as f:
            for file in files:
                f.write(f"{file}\n")

        # list for 减本底
        files = self.file_dict['flat'].copy()
        files.extend(self.file_dict['lamp'])
        files.extend(self.file_dict['std_star'])
        files.extend(self.file_dict['target'])
        files.sort()
        with open(self.dir_process / 'list_for_bias_sub', 'w') as f:
            for file in files:
                f.write(f"{file}\n")

        # list for 合并平场
        files = self.file_dict['flat']
        files = [f"B_{file}" for file in files]
        with open(self.dir_process / 'list_flat', 'w') as f:
            for file in files:
                f.write(f"{file}\n")

        # list for 除平场
        files = self.file_dict['std_star'].copy()
        files.extend(self.file_dict['target'])
        files = [f"B_{file}" for file in files]
        files.sort()
        with open(self.dir_process / 'list_for_flat_div', 'w') as f:
            for file in files:
                f.write(f"{file}\n")
        return None