"""
HSC-SSP数据下载器

@Author: Rui Zhu
@Using code from: https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools
"""
import json
import urllib.request
from pathlib import Path
from loguru import logger
import time

import io
from astropy.io import fits
from astropy.wcs import WCS

from astrokit import CONFIG

__all__ = [
    'my_HSC_account',
    'download_hsc_image', 
    'HSCDownloader'
]

def my_HSC_account(account='Account_HSC'):
    """
    load HSC account information

    Parameters
    ----------
    account : str
        account name in config.yaml, default is 'Account_HSC'
    """
    credential = {
        'account_name': CONFIG[account]['username'], 
        'password': CONFIG[account]['password']
        }
    return credential

def download_hsc_image(
        ra, dec,
        path, 
        bands=['g', 'r', 'i', 'z', 'y'], 
        cutout_size=5, 
        img_type='coadd', 
        account='Account_HSC'):
    """
    Download HSC image cutout

    Parameters
    ----------
    ra : float
        right ascension
    dec : float
        declination
    path : str or Path
        path to save the fits file
    bands : list
        list of bands
    cutout_size : float
        cutout size in arcsec
    img_type : str
        'coadd' (bkg-subtracted) or 'coadd/bkg' (coadd with background)

    """
    from astrokit.externals.HSC_pdr3 import downloadCutout

    primary_header = fits.Header()
    primary_header['RA'] = ra
    primary_header['DEC'] = dec
    primary_header['IMG_TYPE'] = img_type
    primary_header.set(keyword='IMG_TYPE', 
                        value=img_type, 
                        comment="coadd or coadd/bkg")
    primary_header.set(keyword='SIZE', value=cutout_size, 
                       comment='cutout size in arcsec')

    primary_hdu = fits.PrimaryHDU(header=primary_header)
    hdu_list = [primary_hdu]

    for band in bands:
        rect = downloadCutout.Rect.create(
            ra=ra, dec=dec, filter=f"HSC-{band.upper()}", 
            sw=f"{cutout_size/2}arcsec", sh=f"{cutout_size/2}arcsec", 
            mask=True, variance=True, type=img_type
        )
        image = downloadCutout.download(
            rect, 
            user=my_HSC_account(account=account)['account_name'], 
            password=my_HSC_account(account=account)['password'],
        )
        # 从下载的文件中读数据
        hdul = fits.open(io.BytesIO(image[0][1]))

        header_oirg = {}
        header_oirg['primary'] = hdul[0].header
        header_oirg['image'] = hdul[1].header
        header_oirg['mask'] = hdul[2].header
        header_oirg['variance'] = hdul[3].header

        data_oirg = {}
        data_oirg['image'] = hdul[1].data
        data_oirg['mask'] = hdul[2].data
        data_oirg['variance'] = hdul[3].data

        keys_primary = [
            'BGMEAN', 
            'BGVAR', 
            'PSF_ID', 
            'FILTER', 
            'FLUXMAG0', 
        ]
        keys_mask = [
            'MP_BAD', 
            'MP_BRIGHT_OBJECT', 
            'MP_CLIPPED', 
            'MP_CR', 
            'MP_CROSSTALK', 
            'MP_DETECTED', 
            'MP_DETECTED_NEGATIVE', 
            'MP_EDGE', 
            'MP_INEXACT_PSF', 
            'MP_INTRP', 
            'MP_NOT_DEBLENDED', 
            'MP_NO_DATA', 
            'MP_REJECTED', 
            'MP_SAT',
            'MP_SENSOR_EDGE', 
            'MP_SUSPECT', 
            'MP_UNMASKEDNAN'
        ]

        # 创建新的hdu
        for hdu_type in ['image', 'mask', 'variance']:
            header = fits.Header()
            header.update(WCS(header_oirg[hdu_type]).to_header())
            header.set(
                keyword='EXTTYPE',
                value=header_oirg[hdu_type]['EXTTYPE'])
            for key in keys_primary:
                header.set(
                    keyword=key, 
                    value=header_oirg['primary'][key],
                    comment=header_oirg['primary'].comments[key])
            if hdu_type == 'mask':
                for key in keys_mask:
                    header.set(
                        keyword=f"HIERARCH {key}", 
                        value=header_oirg['mask'][key],
                        comment=header_oirg['mask'].comments[key])
            hdu = fits.ImageHDU(
                name=f'{band}_{hdu_type}', 
                data=data_oirg[hdu_type], 
                header=header, 
                )
            hdu_list.append(hdu)
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(path, overwrite=True)

    return None


class HSCDownloader:
    """
    HSC-SSP星表下载器
    """
    def __init__(self, 
                 credential=my_HSC_account(account='Account_HSC'), 
                 output_format="csv", 
                 release_version="pdr3-citus-columnar", 
                 dir_output=Path(CONFIG['PATH_DOWNLOAD']), 
                 email=False,
                 silent=False):
        """
        Parameters
        ----------
        credential : dict
            账号信息
        output_format : str
            输出格式, 默认为csv; 可选csv, csv.gz, sqlite3, fits
        release_version : str
            数据版本, 默认为dr3-citus; 可选pdr1 pdr2 pdr2-citus pdr3 pdr3-citus pdr3-citus-columnar
        dir_output : str
            下载目录
        email : bool
            是否发送邮件
        silent : bool
            是否静默模式

        
        """
        self.credential = credential
        self.api_url = "https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/"
        self.output_format = output_format
        self.release_version = release_version
        self.dir_output = Path(dir_output)
        self.noemail = not email
        self.silent = silent
    
    def sent_request(self, url, post_dict):
        postData = {
            'credential': self.credential,
            'clientVersion': 20190514.1, 
            'nomail': self.noemail
        }
        postData.update(post_dict)
        postData = json.dumps(postData)
        headers = {'Content-type': 'application/json'}
        request = urllib.request.Request(url, postData.encode('utf-8'), headers)
        response = urllib.request.urlopen(request)
        return response
    
    def submit_job(self, path_sql=None, str_sql=None, job_name=None, silent=None):
        """
        提交SQL任务, 可以选择传入SQL文件路径或SQL字符串

        Parameters
        ----------
        path_sql : str
            SQL文件路径
        str_sql : str
            SQL字符串
        job_name : str
            任务名称
        """
        if silent == None:
            silent = self.silent
        
        if (path_sql == None) and (str_sql == None):
            raise ValueError("Please provide path_sql or sql string.")
        if (path_sql != None) and (str_sql != None):
            raise ValueError("Please provide only one of path_sql or sql string.")
        if path_sql == None:
            sql = str_sql
        if str_sql == None:
            with open(path_sql, 'r') as f:
                sql = f.read()

        catalog_job = {
            'sql': sql, 
            'name': job_name,
            'out_format': self.output_format,
            'include_metainfo_to_body': False,
            'release_version': self.release_version,
            }
        url = self.api_url + 'submit'
        response = self.sent_request(url, {'catalog_job': catalog_job})
        job = json.loads(response.read())
        if not silent:
            logger.info(f"Job ID: {job['id']} | "
                        f"Submitted! | "
                        f"Job Name: {job['name']} | "
                        f"HSC Account: {self.credential['account_name']}")
        return job['id']
    
    def query_job(self, job_id):
        url = self.api_url + 'status'
        response = self.sent_request(url, {'id': job_id})
        res = json.loads(response.read())
        return res
    
    def delete_job(self, job_id, silent=None):
        if silent == None:
            silent = self.silent
        url = self.api_url + 'delete'
        response = self.sent_request(url, {'id': job_id})
        if not silent:
            logger.info(f"Job ID: {job_id} | File is deleted on HSC server!")
        return None
    
    def download(self, job_id, output_fname=None, silent=None):
        """
        从网站上下载对应job_id的数据

        Parameters
        ----------
        job_id : str
            任务ID
        fname : str
            下载文件名, 默认为job_id, 不包含后缀
        """
        if silent == None:
            silent = self.silent
        info = self.query_job(job_id)
        if output_fname == None:
            fname = f"{info['name']}.{self.output_format}"
        else:
            fname = output_fname
        if info['status'] == 'done':
            if not silent:
                size = round(info['filesize'] / 1024**2, 3)
                if size > 1024:
                    size = round(size / 1024, 3)
                    size = f"{size} GB"
                else:
                    size = f"{size} MB"
                logger.info(f"Job ID: {job_id} | Downloading result file (size: {size}) ... ")
        else:
            raise ValueError(f"Job {job_id} is not done.")
        
        url = self.api_url + 'download'
        response = self.sent_request(url, {'id': job_id})

        with open(self.dir_output / f"{fname}", 'wb') as f:
            f.write(response.read())
        if not silent:
            logger.info(f"Job ID: {job_id} | Result has been downloaded. | File: {fname}")
    
    def auto_download(self, path_sql=None, str_sql=None, job_name=None, silent=None):
        """
        提交任务并下载数据

        Parameters
        ----------
        path_sql : str
            SQL文件路径
        str_sql : str
            SQL字符串
        job_name : str
            任务名称
        """
        self.is_free = False
        if silent == None:
            silent = self.silent
        st_time = time.time()
        try:
            job_id = self.submit_job(path_sql, str_sql, job_name, silent=True)
            if not silent:
                logger.info(f"Job ID: {job_id} | "
                            f"Job Name: {job_name} | "
                            f"Account: {self.credential['account_name']} | "
                            f"running ...")
            while True:
                res = self.query_job(job_id)
                status = res['status']
                if status == "running":
                    time.sleep(10)
                elif status == "waiting":
                    time.sleep(3)
                else:
                    break
            if status == "done":
                if not silent:
                    logger.info(f"Job ID: {job_id} | Query finished and downloading result ...")
                self.download(job_id, silent=True)
            else:
                logger.error(f"Job ID: {job_id} failed.")
            self.delete_job(job_id, silent=True)
        except:
            logger.error(f"{job_name} something wrong.")
        if not silent:
            cost_time = round(time.time() - st_time, 2)
            if cost_time > 60:
                cost_time /= 60
                logger.success(f"Job ID: {job_id} | Done! | "
                               f"Job Name: {job_name}| "
                               f"cost time: {cost_time} min")
            else:
                logger.success(f"Job ID: {job_id} | Done! | "
                               f"Job Name: {job_name}| "
                               f"cost time: {cost_time} s")
        self.is_free = True
        return None


