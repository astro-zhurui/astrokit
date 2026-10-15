__author__ = 'Rui Zhu'
__email__ = 'zhurui675@gmail.com'

__version__ = '0.1.1'

import sys
import yaml
import socket
from datetime import timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
import importlib
from loguru import logger

hostname = socket.gethostname()
DIR_astrokit = Path(__file__).resolve().parent  # astrokit directory

# 加载配置文件
PATH_config = Path.home() / '.astrokit_config.yaml'
if not PATH_config.exists():
    warning_msg = (
        f"\n[astrokit] Configuration file not found at {PATH_config}."
        "\n[astrokit] Please create a configuration file named '.astrokit_config.yaml' in the home directory."
        "\n[astrokit] You can follow the template provided in 'config_template.yaml'."
    )
    raise FileNotFoundError(warning_msg)
else:
    with open(PATH_config, 'r') as f:
        CONFIG = yaml.safe_load(f)

# Quick Directory
DIR_datasets = DIR_astrokit / 'datasets' / 'data'  # default data directory

for path in [DIR_datasets]:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def _logger_format():
    return (
        "<fg #4169E1>[astrokit]</> "
        "<fg #87CEEB>{time:YYYY-MM-DD HH:mm:ss}</> | "
        "<level>{message}</level>"
    )


def _use_beijing_time(record):
    record["time"] = record["time"].astimezone(timezone(timedelta(hours=8)))
    return True


def set_logger_config():
    """Set logger configuration (timestamps in Beijing time, UTC+8)."""
    logger.remove()
    logger.add(sink=sys.stdout, level="INFO", format=_logger_format(), filter=_use_beijing_time)


# lazy import submodules and Top level functions
__all__ = [
    'datasets', 
    'extinction',
    'ML', 
    'observation',
    'phot', 
    'photoz', 
    'simqso',
    'spec', 
    'toolbox',
    'wrapper'
]

from ._top_level_map import _lazy_map

# Static exports for type checkers. Runtime lookup stays in __getattr__.
# `import x as x` re-exports the name (PEP 484); a bare import would stay private.
if TYPE_CHECKING:
    from .toolbox.email import send_email as send_email
    from .toolbox.calculate import (
        ABmagErr_to_flamErr as ABmagErr_to_flamErr,
        ABmagErr_to_fnuErr as ABmagErr_to_fnuErr,
        ABmag_to_flam as ABmag_to_flam,
        ABmag_to_fnu as ABmag_to_fnu,
        NMAD as NMAD,
        binned_stats as binned_stats,
        crack as crack,
        flamErr_to_ABmagErr as flamErr_to_ABmagErr,
        flamErr_to_fnuErr as flamErr_to_fnuErr,
        flam_to_ABmag as flam_to_ABmag,
        flam_to_fnu as flam_to_fnu,
        fnuErr_to_ABmagErr as fnuErr_to_ABmagErr,
        fnuErr_to_flamErr as fnuErr_to_flamErr,
        fnu_to_ABmag as fnu_to_ABmag,
        fnu_to_flam as fnu_to_flam,
    )
    from .toolbox.plot import (
        add_colorbar_ax as add_colorbar_ax,
        add_text as add_text,
        bold_axis as bold_axis,
        imshow as imshow,
        plot_SDSS_spectrum as plot_SDSS_spectrum,
        plot_aperture_photometry_growth_curve as plot_aperture_photometry_growth_curve,
        plot_scatter_distribution as plot_scatter_distribution,
        plot_stamps as plot_stamps,
        savefig as savefig,
        show_colors as show_colors,
    )
    from .toolbox.utils import (
        clear as clear,
        find_process_by_name as find_process_by_name,
        fits2df as fits2df,
        pandas_show_all_columns as pandas_show_all_columns,
        print_directory_tree as print_directory_tree,
        read as read,
        read_wcs as read_wcs,
        run_command_in_terminal as run_command_in_terminal,
        sec_to_hms as sec_to_hms,
        set_mpl_style as set_mpl_style,
        show_device_info as show_device_info,
        show_internet_speed as show_internet_speed,
        use_svg_display as use_svg_display,
        value_to_KVD_string as value_to_KVD_string,
    )

def build_lazy_subfuncs():
    """
    将部分挑选的函数从子模块中导入到当前命名空间，
    以便用户可以直接使用这些函数，而不需要导入子模块。
    """
    registry = {}
    for submod, names in _lazy_map.items():
        for name in names:
            if name in registry:
                raise ValueError(f"函数名冲突: {name} 同时出现在 {registry[name][0]} 和 {submod}")
            registry[name] = (submod, name)

    return registry

_lazy_subfuncs = build_lazy_subfuncs()

def __getattr__(name: str) -> Any:
    # importlib.import_module is typed as ModuleType. Without an explicit
    # return type, checkers treat every lazy attribute as a module and reject
    # calls such as ak.set_mpl_style(...).
    if name in __all__:
        submod = importlib.import_module(f".{name}", __package__)
        globals()[name] = submod
        return submod
    elif name in _lazy_subfuncs:
        submod_path, obj_name = _lazy_subfuncs[name]
        submod = importlib.import_module(f".{submod_path}", __package__)
        obj = getattr(submod, obj_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__} has no attribute {name}")

def _auto_import_all(package_globals, package_name, package_file):
    """
    自动导入 package_file 所在目录下所有 py 文件中的 __all__ 内容，
    并更新传入模块的 globals() 和 __all__

    参数:
    - package_globals: 调用方模块的 globals()
    - package_name: 调用方模块的 __name__
    - package_file: 调用方模块的 __file__
    """
    all_exports = []

    package_path = Path(package_file).parent

    for py_file in package_path.glob('*.py'):
        if py_file.name == '__init__.py':
            continue

        module_name = py_file.stem  # stem = 文件名去掉 .py
        module = importlib.import_module(f'.{module_name}', package=package_name)

        if hasattr(module, '__all__'):
            names = getattr(module, '__all__')
            all_exports.extend(names)

            package_globals.update({name: getattr(module, name) for name in names})

    package_globals['__all__'] = all_exports

# default logger configuration (Beijing time, UTC+8)
logger.remove()
logger.add(sink=sys.stdout, level="INFO", format=_logger_format(), filter=_use_beijing_time)