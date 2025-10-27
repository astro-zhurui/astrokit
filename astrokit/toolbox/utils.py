"""
Toolbox for save the useful functions

@author: Rui Zhu  
@creation time: 2022-11-29
"""
from pathlib import Path
import pandas as pd
import subprocess
import psutil
import time
from loguru import logger
from IPython.display import clear_output
import warnings
import os

from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
from astropy.table import Table

__all__ = [
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
    "show_internet_speed"
]

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


def fits2df(path_fits):
    """
    读取fits中的table, 并转换成pandas的DataFrame
    """
    tbl = Table.read(path_fits, character_as_bytes=False)
    df = tbl.to_pandas()
    return df

def read(path, no_warnings=True):
    """
    读取fits表格
    """
    if no_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tbl = Table.read(path, character_as_bytes=False)
    else:
        tbl = Table.read(path, character_as_bytes=False)
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
