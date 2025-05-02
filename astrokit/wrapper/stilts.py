"""
A wrapper for the STILTS command line tool.

@ Author: Rui Zhu
@ Date: 2024-10-13
@ STILTS website: https://www.star.bristol.ac.uk/mbt/stilts/
"""
from pathlib import Path
import time
import subprocess
from loguru import logger

__all__ = [
    "tcopy",  # https://www.star.bristol.ac.uk/mbt/stilts/sun256/tcopy-usage.html
    "tmatch2",  # https://www.star.bris.ac.uk/~mbt/stilts/sun256/tmatch2.html
    "tskymatch2", 
    "cdsskymatch",
    "tapskymatch",
]

def tcopy(
        path_input_catalog, 
        path_output_catalog, 
        silence=False,
        ):
    st = time.time()
    path_input_catalog = Path(path_input_catalog)
    path_output_catalog = Path(path_output_catalog)
    args = f"stilts -verbose tcopy"
    args += f" in={str(path_input_catalog)} out={str(path_output_catalog)}"
    if silence:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    else:
        stdout = None
        stderr = None
    process = subprocess.run(
        args=args, 
        cwd=path_input_catalog.parent, 
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True, 
        shell=True, 
    )
    if not silence:
        if process.returncode == 0 & path_output_catalog.exists():
            logger.success(f"Success! | Cost Time: {time.time()-st:.2f}s")
        else:
            logger.error(f"Failed! | Cost Time: {time.time()-st:.2f}s")
    return None


def tmatch2(
        path_cat_left,
        path_cat_right,
        path_cat_output,
        coord_name_left, 
        coord_name_right,
        sep=1.0, 
        find='best',
        join='1and2',
        n_cpu=None, 
        progress='log', 
        silence=False,
):
    """
    Performs a crossmatch of two tables based on the proximity of sky positions.

    Website: https://www.star.bris.ac.uk/~mbt/stilts/sun256/tmatch2.html

    Parameters
    ----------
    path_cat_left : str | Path
        The path to the left catalog.
    path_cat_right : str | Path
        The path to the right catalog.
    path_cat_output : str | Path
        The path to the output catalog.
    coord_name_left : tuple, optional
        The column names of the coordinates in the left catalog.
    coord_name_right : tuple, optional
        The column names of the coordinates in the right catalog.
    sep : float, optional
        The maximum separation (units: arcseconds) between the two objects. The default is 1.0.
    find : str, optional  
        The method used to find the closest match. The default is 'best'.  
        'all': All matches. Every match between the two tables is included in the result.  
        'best': 以对称的方式获得最佳匹配  
        'best1': Best match for each Table 1 row. 
        For each row in table 1, only the best match from table 2 will appear in the result. 
        Each row from table 1 will appear a maximum of once in the result, 
        but rows from table 2 may appear multiple times.  
        'best2': Best match for each Table 2 row. 
        For each row in table 2, only the best match from table 1 will appear in the result. 
        Each row from table 2 will appear a maximum of once in the result, 
        but rows from table 1 may appear multiple times.
    join : str, optional
        Determines which rows are included in the output table. The default is '1and2'.
    n_cpu: str, optional
        The number of CPUs used to run the command. 
        'None': parallel smaller than 6 CPUs. # Default
        'all': parallel all CPUs.
        'n_cpu': parallel n CPUs.
    progress : str, optional
        Determines whether information on progress of the match should be output 
        to the standard error stream as it progresses. 
        For lengthy matches this is a useful reassurance and can give guidance 
        about how much longer it will take. 
        It can also be useful as a performance diagnostic.
        'none': no progress is shown
        'log': progress information is shown
        'time': progress information and some time profiling information is shown
        'profile': progress information and limited time/memory profiling information are shown

    silence : bool, optional
        Whether to suppress the output information. The default is False.
    """
    st = time.time()
    path_cat_left = Path(path_cat_left)
    path_cat_right = Path(path_cat_right)
    path_cat_output = Path(path_cat_output)

    if path_cat_left.exists() & path_cat_right.exists():
        if not silence:
            logger.info(f"Start Matching '{path_cat_left.name}' and '{path_cat_right.name}' ...")
    else:
        raise FileNotFoundError(f"Input File not found!")
    
    args = f"stilts -verbose tmatch2"
    args += f" in1={str(path_cat_left)} in2={str(path_cat_right)}"
    args += f" out={str(path_cat_output)}"
    args += f" values1='{coord_name_left[0]} {coord_name_left[1]}'"
    args += f" values2='{coord_name_right[0]} {coord_name_right[1]}'"
    args += f" matcher=sky"
    args += f" params='{sep}'"
    args += f" find={find}"
    args += f" join={join}"
    args += f" progress={progress}"
    if n_cpu is None:
        args += f" runner=parallel"
    elif n_cpu == "all":
        args += f" runner=parallel-all"
    else:
        args += f" runner=parallel{n_cpu}"

    if silence:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    else:
        stdout = None
        stderr = None

    # 执行命令
    process = subprocess.run(
        args=args, 
        cwd=path_cat_output.parent, 
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True, 
        shell=True, 
    )
    if not silence:
        if process.returncode == 0 & path_cat_output.exists():
            logger.success(f"Success! | Cost Time: {time.time()-st:.2f}s")
        else:
            logger.error(f"Failed! | Cost Time: {time.time()-st:.2f}s")
    return None


def tskymatch2(
        path_cat_left,
        path_cat_right,
        path_cat_output,
        coord_name_left, 
        coord_name_right,
        sep=1.0, 
        find='best',
        join='1and2',
        n_cpu=None, 
        silence=False,
):
    """
    Performs a crossmatch of two tables based on the proximity of sky positions.

    Website: https://www.star.bris.ac.uk/~mbt/stilts/sun256/tskymatch2.html

    Parameters
    ----------
    path_cat_left : str | Path
        The path to the left catalog.
    path_cat_right : str | Path
        The path to the right catalog.
    path_cat_output : str | Path
        The path to the output catalog.
    coord_name_left : tuple, optional
        The column names of the coordinates in the left catalog.
    coord_name_right : tuple, optional
        The column names of the coordinates in the right catalog.
    sep : float, optional
        The maximum separation (units: arcseconds) between the two objects. The default is 1.0.
    find : str, optional  
        The method used to find the closest match. The default is 'best'.  
        'all': All matches. Every match between the two tables is included in the result.  
        'best': 以对称的方式获得最佳匹配  
        'best1': Best match for each Table 1 row. 
        For each row in table 1, only the best match from table 2 will appear in the result. 
        Each row from table 1 will appear a maximum of once in the result, 
        but rows from table 2 may appear multiple times.  
        'best2': Best match for each Table 2 row. 
        For each row in table 2, only the best match from table 1 will appear in the result. 
        Each row from table 2 will appear a maximum of once in the result, 
        but rows from table 1 may appear multiple times.
    join : str, optional
        Determines which rows are included in the output table. The default is '1and2'.
    n_cpu: str, optional
        The number of CPUs used to run the command. 
        'None': parallel smaller than 6 CPUs. # Default
        'all': parallel all CPUs.
        'n_cpu': parallel n CPUs.
    silence : bool, optional
        Whether to suppress the output information. The default is False.
    """
    st = time.time()
    path_cat_left = Path(path_cat_left)
    path_cat_right = Path(path_cat_right)
    path_cat_output = Path(path_cat_output)

    if path_cat_left.exists() & path_cat_right.exists():
        if not silence:
            logger.info(f"Start Matching '{path_cat_left.name}' and '{path_cat_right.name}' ...")
    else:
        raise FileNotFoundError(f"Input File not found!")
    
    args = f"stilts -verbose tskymatch2"
    args += f" in1={str(path_cat_left)} in2={str(path_cat_right)}"
    args += f" out={str(path_cat_output)}"
    args += f" ra1={coord_name_left[0]} dec1={coord_name_left[1]}"
    args += f" ra2={coord_name_right[0]} dec2={coord_name_right[1]}"
    args += f" error={sep}"
    args += f" find={find}"
    args += f" join={join}"
    if n_cpu is None:
        args += f" runner=parallel"
    elif n_cpu == "all":
        args += f" runner=parallel-all"
    else:
        args += f" runner=parallel{n_cpu}"

    if silence:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    else:
        stdout = None
        stderr = None

    # 执行命令
    process = subprocess.run(
        args=args, 
        cwd=path_cat_output.parent, 
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True, 
        shell=True, 
    )
    if not silence:
        if process.returncode == 0 & path_cat_output.exists():
            logger.success(f"Success! | Cost Time: {time.time()-st:.2f}s")
        else:
            logger.error(f"Failed! | Cost Time: {time.time()-st:.2f}s")
    return None




def cdsskymatch(
        path_input_catalog, 
        path_output_catalog,
        cds_catalog_name, 
        ra='ra', dec='dec', 
        sep=1.0, 
        find='best',
        blocksize=50000, 
        timeout=None, 
        silence=False,
        ):
    """
    Online crossmatch with VizieR tables and the SIMBAD database.

    Website: https://www.star.bristol.ac.uk/mbt/stilts/sun256/cdsskymatch.html

    Parameters
    ----------
    path_input_catalog : str | Path
        The path to the input catalog.
    path_output_catalog : str | Path
        The path to the output catalog.
    cds_catalog_name : str
        The name of the CDS catalog.
    ra : str, optional
        The name of the right ascension column. The default is 'ra'.
    dec : str, optional
        The name of the declination column. The default is 'dec'.
    sep : float, optional
        The maximum separation (units: arcseconds) between the two objects. The default is 1.0.
    """
    st = time.time()
    path_input_catalog = Path(path_input_catalog)

    if not silence:
        logger.info(f"Start Matching '{path_input_catalog.name}' from '{cds_catalog_name}' ...")

    args = f"stilts -verbose cdsskymatch"

    # 添加命令参数
    if path_input_catalog.exists():
        args += f" in={str(path_input_catalog)}"
    else:
        raise FileNotFoundError(f"Input File not found: {path_input_catalog}")
    
    args += f" out={str(path_output_catalog)}"
    args += f" cdstable={cds_catalog_name}"
    args += f" ra={ra} dec={dec}"
    args += f" radius={sep}"
    args += f" find={find}"
    args += f" blocksize={blocksize}"

    if silence:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    else:
        stdout = None
        stderr = None

    # 执行命令
    process = subprocess.run(
        args=args, 
        cwd=path_input_catalog.parent, 
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True, 
        shell=True, 
        timeout=timeout
    )
    if not silence:
        if process.returncode == 0 & path_output_catalog.exists():
            logger.success(f"Success! | Cost Time: {time.time()-st:.2f}s")
        else:
            logger.error(f"Failed! | Cost Time: {time.time()-st:.2f}s")

    return None


def tapskymatch(
        path_input_catalog, 
        path_output_catalog,
        url_tap, 
        table_tap, 
        coord_name_local, 
        coord_name_tap,
        blocksize=5000, 
        find='all', 
        sep=1.0, 
        timeout=None, 
        silence=False,
        ):
    """
    Crossmatch of a local table with one held in a remote TAP service.

    Website: https://www.star.bristol.ac.uk/mbt/stilts/sun256/tapskymatch.html

    Parameters
    ----------
    blocksize : int, optional
        The number of rows to be sent to the TAP service in each query. The default is 5000.
    find : str, optional
        'all': All matches.  
        'best': Matched rows, best remote row for each input row  
        'each': One row per input row, contains best remote match or blank  
        'each-dist': One row per input row, column giving distance only for best match
    sep : float, optional
        The maximum separation (units: arcseconds) between the two objects. The default is 1.0.
    """
    st = time.time()
    path_input_catalog = Path(path_input_catalog)
    path_output_catalog = Path(path_output_catalog)
    
    if path_input_catalog.exists():
        logger.info(f"Start Matching '{path_input_catalog.name}' from '{url_tap}' ...")
    else:
        raise FileNotFoundError(f"Input File not found: {path_input_catalog}")
    
    args = f"stilts -verbose tapskymatch"
    args += f" in={str(path_input_catalog)}"
    args += f" out={str(path_output_catalog)}"
    args += f" tapurl={url_tap} taptable={table_tap}"
    args += f" inlon={coord_name_local[0]} inlat={coord_name_local[1]}"
    args += f" taplon={coord_name_tap[0]} taplat={coord_name_tap[1]}"
    args += f" blocksize={blocksize} find={find}"
    sr = round(sep * (1/3600), 6)
    args += f" sr={sr}"

    if silence:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    else:
        stdout = None
        stderr = None

    # 执行命令
    process = subprocess.Popen(
        args=args, 
        cwd=path_input_catalog.parent, 
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True, 
        shell=True, 
    )
    pid = process.pid
    if not silence:
        logger.info(f"Start Matching | PID: {pid}")
    returncode = process.wait(timeout=timeout)

    if not silence:
        if returncode == 0 & path_output_catalog.exists():
            logger.success(f"Success! | Cost Time: {time.time()-st:.2f}s")
        else:
            logger.error(f"Failed! | Cost Time: {time.time()-st:.2f}s")
    return None


