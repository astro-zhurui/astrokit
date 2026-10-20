"""
Toolbox for save the useful functions

@author: Rui Zhu  
@creation time: 2022-11-29
"""
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import psutil
import time
from loguru import logger
from IPython.display import clear_output
import warnings
import os
import matplotlib as mpl

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
from astropy.table import Table, MaskedColumn

__all__ = [
    "set_mpl_style",
    "show_device_info", 
    "clear",
    "pandas_show_all_columns",
    "use_svg_display",
    "run_command",
    "run_command_in_terminal",
    "find_process_by_name",
    "value_to_KVD_string",
    "fits2df",
    "read", 
    "read_wcs", 
    "print_directory_tree", 
    "sec_to_hms", 
    "show_internet_speed", 
    "replace_list_items",
    "get_device_name",
]
def set_mpl_style(
    fontsize=12,
    title_size=14,
    label_size=14,
    tick_size=12,
    legend_size=11,
    font="serif",
    use_tex=False,
    *,
    rc=None,
):
    """Apply the compact ApJ-style defaults to future matplotlib artists.

    Parameters
    ----------
    fontsize : float, optional
        Base and annotation font size in points. Default is 12.
    title_size, label_size : float, optional
        Axes title and label sizes in points. Defaults are 14.
    tick_size, legend_size : float, optional
        Tick and legend sizes in points. Defaults are 12 and 11.
    font : str or sequence of str, optional
        Font family. Default is ``"serif"``, with Times New Roman,
        STIXGeneral and DejaVu Serif as ordered fallbacks.
    use_tex : bool, optional
        Render text with LaTeX. Default is False; math uses STIX.
    rc : mapping, optional
        Additional matplotlib rcParams, applied after the named parameters.
        All settings are validated before updating the global configuration.

    Notes
    -----
    Sets inward ticks on four sides, blue/orange-first colors, 0.7 pt axes,
    1.2 pt lines, a white background, 450 dpi exports and TrueType PDF/PS
    fonts. Figure size, panel layout and data-specific colors are set per plot.
    Existing artists retain their settings. Explicit arguments override the
    defaults; the original seven positional arguments remain supported.

    Examples
    --------
    >>> import astrokit as ak
    >>> ak.set_mpl_style()
    >>> ak.set_mpl_style(label_size=16, rc={"lines.linewidth": 1.5})
    """
    params = {
        "font.family": font,
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": fontsize,
        "mathtext.fontset": "stix",
        "text.usetex": use_tex,
        "text.color": "black",
        "figure.facecolor": "white",
        "figure.dpi": 140,
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "axes.unicode_minus": True,
        "axes.prop_cycle": mpl.cycler(color=[
            "#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#56B4E9", "#E69F00", "#F0E442", "#333333",
        ]),
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.2,
        "legend.fontsize": legend_size,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 450,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
    }
    if rc is not None:
        params.update(rc)
    mpl.rcParams.update(mpl.RcParams(params))

def show_internet_speed(interval=1):
    """Display real-time internet speed for all network interfaces."""
    try:
        while True:
            # 获取当前网卡统计
            net1 = psutil.net_io_counters(pernic=True)
            time.sleep(interval)
            net2 = psutil.net_io_counters(pernic=True)

            # 清屏
            os.system('clear')  # Linux/macOS，Windows 用 'cls'

            # 显示每个网卡速度
            for nic in net1:
                sent = (net2[nic].bytes_sent - net1[nic].bytes_sent) / (1024**2)
                recv = (net2[nic].bytes_recv - net1[nic].bytes_recv) / (1024**2)
                print(f"{nic}: Up {sent/interval:.2f} MB/s | Down {recv/interval:.2f} MB/s")

            print("-" * 50)

    except KeyboardInterrupt:
        print("\nStopped by user")

def show_device_info():
    """
    显示当前设备信息
    """
    import socket
    import psutil
    import GPUtil

    hostname = socket.gethostname()
    n_cpu = psutil.cpu_count(logical=True)
    try:
        gpus = GPUtil.getGPUs()
    except ValueError:
        gpus = None

    print(f"==> [Device Name] {hostname}")
    print(f"==> [CPU Info] {n_cpu} logical cores")
    print(f"==> [Memory Info]:")
    print(f"    Total Memory: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    print(f"    Used Memory: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
    print(f"    Available Memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
    print(f"    Memory Usage: {psutil.virtual_memory().percent}%")
    if gpus is None:
        print(f"==> [GPU Info]: No GPU found")
    else:
        print(f"==> [GPU Info]:")
        for gpu in gpus:
            print(
                f"    GPU ID: {gpu.id}, "
                f"Name: {gpu.name}, "
                f"Load: {gpu.load*100:.1f}%, "
                f"Total Memory: {gpu.memoryTotal}MB, "
                f"Memory Used: {gpu.memoryUsed}MB"
            )
    return None

def clear():
    clear_output()
    return None

def pandas_show_all_columns():
    """
    设置pandas显示所有列
    """
    pd.set_option('display.max_columns', None)
    return None

def use_svg_display():
    """
    将matplotlib在jupyter里的显示图片格式设置为svg
    """
    from matplotlib_inline import backend_inline
    backend_inline.set_matplotlib_formats('svg')

def run_command(cmd, dir_work, print_output=True, timeout=None):
    """
    Run a shell command in a specific directory, printing output in real-time.
    
    Parameters
    ----------
    cmd : str
        The command to run
    dir_work : str
        The directory to run the command in
    timeout : float, optional
        Timeout for the command in seconds, default is None
    
    Returns
    -------
    int
        The return code of the command
        - 0   : success
        - 127 : command not found
        - -1  : timeout
    """
    try:
        process = subprocess.Popen(
            cmd,
            cwd=dir_work,
            shell=True,                 # 通过 shell 执行
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # 合并 stderr，避免线程阻塞
            text=True,
            bufsize=1,                  # 行缓冲
        )

        if print_output:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.strip())

        returncode = process.wait(timeout=timeout)

        return returncode

    except subprocess.TimeoutExpired:
        logger.error(f"Time out!")
        process.terminate()
        return -1
    
    finally:
        if process.stdout:
            process.stdout.close()

def run_command_in_terminal(cmd) -> None:
    """Run a shell command line in terminal"""
    # AppleScript脚本
    applescript = f"""
    tell application "Terminal"
        if not (exists window 1) then
            do script "{cmd}"
        else
            do script "{cmd}" in window 1
        end if
        activate
    end tell
    """

    # 使用subprocess执行AppleScript脚本
    subprocess.run(['osascript', '-e', applescript], check=True)
    return None


def find_process_by_name(process_name):
    """search the process name, if this porcess is running, return True, else False"""
    
    for process in psutil.process_iter(attrs=["pid", "name"]):
        if process.info["name"] == process_name:
            return True
    return False

def value_to_KVD_string(value) -> str|None:
    """
    将int, float, couple, None等数据类型转换成上古软件配置文件常用的
    keyword, value, description(KVD)中的value字符串
    """

    if isinstance(value, int|float|str|Path):
        string = f"{value}"
    if isinstance(value, type(None)):
        string = None
    if isinstance(value, list|tuple):
        string = str(value).strip("()[]")
        string = string.replace("'", "")

    return string

def fits2df(path_fits, columns=None, n_rows=None, hdu_name=None):
    """Read a FITS table into pandas without parsing column units.

    Parameters
    ----------
    path_fits : str or pathlib.Path
        FITS file, including gzip-compressed files.
    columns : sequence of str or str, optional
        Columns to read, in output order. Defaults to all columns.
    n_rows : int, optional
        Number of leading rows to return. Defaults to all rows.
    hdu_name : int or str, optional
        Table extension. Defaults to the first table.

    Returns
    -------
    pandas.DataFrame
        Native-endian values with decoded strings. Undefined logical values
        become pandas.NA. FITS units are not retained or converted.
    """
    with fits.open(path_fits, memmap=False, character_as_bytes=False) as hdul:
        if hdu_name is None:
            hdu = next((item for item in hdul if isinstance(item, (fits.BinTableHDU, fits.TableHDU))), None)
            if hdu is None:
                raise ValueError("No FITS table found")
        else:
            hdu = hdul[hdu_name]
        total_rows = hdu.header["NAXIS2"]
        if n_rows is None:
            n_rows = total_rows
        elif not 0 <= n_rows <= total_rows:
            raise ValueError(f"n_rows must be between 0 and {total_rows}, got {n_rows}")
        names = list(hdu.columns.names if columns is None else np.atleast_1d(columns))
        missing = sorted(set(names) - set(hdu.columns.names))
        if missing:
            raise KeyError(f"Unknown column(s): {missing}")
        table = Table()
        for name in names:
            column = hdu.columns[name]
            if str(column.format).endswith("L"):
                values = np.recarray.field(hdu.data, name)[:n_rows]
                valid = (values == ord("T")) | (values == ord("F"))
                table[name] = MaskedColumn(values == ord("T"), mask=~valid)
                continue
            values = hdu.data[name][:n_rows]
            if column.null is not None:
                table[name] = MaskedColumn(values, mask=values == column.null)
            elif values.dtype.kind in "fc":
                table[name] = MaskedColumn(values, mask=np.isnan(values))
            elif values.dtype.kind in "SU":
                values = np.char.rstrip(values)
                table[name] = MaskedColumn(values, mask=values == (b"" if values.dtype.kind == "S" else ""))
            else:
                table[name] = values
        df = table.to_pandas()
        for name in names:
            if str(hdu.columns[name].format).endswith("L") and np.any(table[name].mask):
                df[name] = pd.array(df[name], dtype="boolean")
        return df

def read(path, n_rows=None, columns=None, hdu_name=None, fill=False, show_time=False):
    """
    Read rows from a large FITS binary table using memory mapping.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the FITS file.
    n_rows : int or None, optional
        Number of rows to read from the beginning of the table. If None,
        all rows are read.
    columns : sequence of str, str, or None, optional
        Column names to include in the returned table. If None, all columns
        are included.
    fill : bool, optional
        If True, requested columns that are not present in the FITS table are
        included as fully masked columns. If False, missing requested columns
        raise KeyError.
    show_time : bool, optional
        If True, print the elapsed read time after finishing. Default is False.

    Returns
    -------
    astropy.table.Table
        Table containing the requested rows and columns. Numeric columns are
        backed by memory-mapped arrays when possible; FITS logical columns are
        converted from T/F bytes to boolean arrays, and FITS character columns
        are decoded to unicode strings with trailing FITS padding removed.

    Raises
    ------
    ValueError
        If n_rows is negative or larger than the number of rows in the table.
    KeyError
        If fill is False and any requested column name is not present in the
        FITS table.
    """
    def _decode_fits_text(values):
        if values.dtype.kind == "U":
            return np.char.rstrip(values)

        if values.dtype.kind == "S":
            try:
                decoded = np.char.decode(values, "ascii")
            except UnicodeDecodeError:
                decoded = np.char.decode(values, "utf-8")
            return np.char.rstrip(decoded)

        return values

    st = time.time()
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        if hdu_name is not None:
            hdu = hdul[hdu_name]
        else:
            hdu = hdul[1]
        total_rows = hdu.header["NAXIS2"]
        if n_rows is None:
            n_rows = total_rows
        elif not 0 <= n_rows <= total_rows:
            raise ValueError(f"n_rows must be between 0 and {total_rows}, got {n_rows}")

        column_names = list(hdu.columns.names if columns is None else np.atleast_1d(columns))
        missing = sorted(set(column_names) - set(hdu.columns.names))
        if missing and not fill:
            raise KeyError(f"Unknown column(s): {missing}")

        data_offset = hdu.fileinfo()["datLoc"]
        row_dtype = hdu.columns.dtype.newbyteorder(">")
        rows = np.memmap(path, mode="r", dtype=row_dtype, offset=data_offset, shape=(n_rows,))
        column_by_name = {column.name: column for column in hdu.columns}
        table_data = {}

        for name in column_names:
            column = column_by_name.get(name)
            if column is None:
                table_data[name] = MaskedColumn(
                    data=np.empty(n_rows, dtype=float),
                    mask=np.ones(n_rows, dtype=bool),
                    name=name,
                )
            elif str(column.format).endswith("L"):
                valid = (rows[name] == ord("T")) | (rows[name] == ord("F"))
                table_data[name] = MaskedColumn(rows[name] == ord("T"), mask=~valid)
            elif rows.dtype[name].kind in {"S", "U"} or str(column.format).endswith("A"):
                table_data[name] = _decode_fits_text(rows[name])
            else:
                table_data[name] = rows[name]

        tbl = Table(table_data, copy=False)

    if show_time:
        print(f"Read {path.name}: {sec_to_hms(time.time() - st)}")

    return tbl

def read_wcs(header):
    """
    构造 WCS 对象，并局部屏蔽 FITSFixedWarning。
    其他 warning 类型仍然会正常显示。
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FITSFixedWarning)
        return WCS(header)


def print_directory_tree(path, level=0, show_hidden=False, indent="", current_level=0):
    """
    打印目录树

    Parameters
    ----------
    path : str or Path
        目录路径
    level : int, optional
        打印几层目录树, 默认为0, 即打印所有层级
    show_hidden : bool, optional
        是否显示隐藏文件, 默认为False
    indent : str, optional (无需传入，内部使用)
        缩进字符, 默认为空
    current_level : int, optional (无需传入，内部使用)
        当前目录层级, 默认为0
    """

    if level != 0 and current_level >= level:
        return None
    
    # 获取目录下的所有文件和子目录，并根据条件过滤隐藏文件
    items = [item for item in Path(path).iterdir() if show_hidden or not item.name.startswith('.')]
    
    # 排序，目录在前，文件在后
    items.sort(key=lambda x: (x.is_file(), x.name))

    for i, item in enumerate(items):
        # 判断是否是最后一个元素
        is_last = (i == len(items) - 1)
        
        # 打印当前元素
        prefix = '└── ' if is_last else '├── '
        if item.is_dir():
            print(f"{indent}{prefix}{item.name}/")
            # 递归打印子目录
            new_indent = indent + ("    " if is_last else "│   ")
            print_directory_tree(path=item, level=level, 
                                 show_hidden=show_hidden, 
                                 indent=new_indent, 
                                 current_level=current_level + 1)
        else:
            print(f"{indent}{prefix}{item.name}")

def sec_to_hms(seconds, str_format=True):
    """
    将秒数转换为时分秒

    Parameters
    ----------
    seconds : int
        秒数
    str_format : bool, optional
        是否返回字符串格式, 默认为True
    """
    h, remainder = divmod(seconds, 3600)  # 计算小时
    m, s = divmod(remainder, 60)         # 计算分钟和秒
    if str_format:
        if (h == 0) and (m == 0):
            return f"{s:.2f} s"
        elif h == 0:
            return f"{m:.0f} min, {s:.2f} s"
        else:
            return f"{h:.0f} h, {m:.0f} min, {s:.2f} s"
    else:
        return h, m, s

def replace_list_items(columns: dict, ls: list):
    """
    Replace items in a list based on a mapping dictionary.

    Parameters
    ----------
    columns : dict
        A dictionary mapping old items to new items.
    ls : list
        The list in which to replace items.

    Returns
    -------
    list
        The modified list with items replaced.
    """
    for key, value in columns.items():
        ls = [value if x == key else x for x in ls]
    return ls


def get_device_name() -> str:
    """
    返回当前程序运行的设备名称（主机名）。

    Returns
    -------
    str
        当前设备的主机名。
    """
    import socket
    return socket.gethostname()

