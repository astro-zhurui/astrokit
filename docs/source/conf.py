# docs/source/conf.py

# -- Path setup ---------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # 让 Sphinx 找到你的包

# -- Project information ------------------------------------------
project = 'AstroKit'
author = 'Rui Zhu'
release = '0.1.0'  # 项目版本号 (建议符合 PEP 440)

# -- General configuration ----------------------------------------
extensions = [
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

# -- MyST (Markdown) configuration -------------------------------
myst_enable_extensions = [
    "colon_fence",       # 支持 ::: 块 (admonitions)
    "linkify",           # 自动链接 http
]
myst_heading_anchors = 3   # h1~h3 标题自动生成锚点 (方便链接跳转)

# -- HTML output --------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for HTML theme ---------------------------------------
html_theme_options = {
    'navigation_depth': 3,   # 目录树最大深度
    'collapse_navigation': False,
    'titles_only': False,
}