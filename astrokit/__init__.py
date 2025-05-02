__author__ = 'Rui Zhu'
__email__ = 'zhurui675@gmail.com'

__version__ = '0.1.0'

import sys
import yaml
from pathlib import Path
import importlib

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
DIR_data = DIR_astrokit / 'datasets' / 'data'  # default data directory
DIR_download = Path(CONFIG['PATH_DOWNLOAD'])  # default download directory

for path in [DIR_data]:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# Add Project Tools' Path
for path in CONFIG['PATH_PROJECT_TOOL']:
    if Path(path).exists():
        sys.path.append(path)

def set_logger_config():
    """
    Set logger configuration.
    """
    from loguru import logger
    logger.remove()
    logger.add(
        sink=sys.stdout, level="INFO", 
        format=("<fg #4169E1>[astrokit]</> "
                "<fg #87CEEB>{time:YYYY-MM-DD HH:mm:ss}</> "
                "| <level>{message}</level>")
        )

# lazy import submodules and Top level functions
__all__ = [
    'datasets', 
    'extinction',
    'ML', 
    'observation',
    'phot', 
    'photoz', 
    'spec', 
    'toolbox',
    'wrapper'
]

from ._top_level_map import _lazy_map

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

def __getattr__(name):
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
