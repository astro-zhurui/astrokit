"""Photometric-redshift utilities with optional modules loaded on demand."""
from importlib import import_module

EXPORTS = {
    'utils': (
        'cal_RMSE', 'cal_NMAD', 'cal_bias', 'cal_outlier_fraction',
        'cal_R2', 'plot_zspec_zphot',
    ),
    'colorz': ('ColorZ', 'run'),
    'template_fitting': ('save_template', 'SWIRETemplates'),
}
__all__ = [name for names in EXPORTS.values() for name in names]


def __getattr__(name):
    for module, names in EXPORTS.items():
        if name in names:
            value = getattr(import_module(f'.{module}', __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(__all__))
