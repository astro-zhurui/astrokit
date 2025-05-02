"""
Toolbox for save the useful functions

@author: Rui Zhu  
@creation time: 2022-11-29
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import psutil
from IPython.display import clear_output

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


__all__ = [
    "clear",
    "pandas_show_all_columns",
    "use_svg_display",
    "run_cmd_in_terminal",
    "find_process_by_name",
    "value_to_KVD_string",
    "fits2df",
    "print_directory_tree", 
    "sec_to_hms"
]

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


def run_cmd_in_terminal(cmd) -> None:
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
