_lazy_map = {
    'toolbox.calculate': [
        'fnu_to_flam', 'flam_to_fnu',
        'fnu_to_ABmag', 'ABmag_to_fnu',
        'flam_to_ABmag', 'ABmag_to_flam',
        'ABmagErr_to_fnuErr', 'fnuErr_to_ABmagErr',
        'fnuErr_to_flamErr', 'flamErr_to_fnuErr',
        'ABmagErr_to_flamErr', 'flamErr_to_ABmagErr',
        'crack', 'binned_stats', 'NMAD',
    ],
    'toolbox.plot': [
        'show_colors', 'add_text', 'savefig',
        'bold_axis', 'add_colorbar_ax',
        'plot_scatter_distribution', 'plot_aperture_photometry_growth_curve',
        'plot_stamps', 'plot_SDSS_spectrum', 'imshow',
    ],
    'toolbox.utils': [
        'clear', 'pandas_show_all_columns', 'use_svg_display',
        'run_cmd_in_terminal', 'find_process_by_name',
        'value_to_KVD_string', 'fits2df', 'print_directory_tree',
        'sec_to_hms',
    ],
}
