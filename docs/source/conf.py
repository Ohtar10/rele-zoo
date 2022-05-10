# Configuration file for the Sphinx documentation builder.
import sys
import os
from setuptools import find_packages

sys.path.insert(0, os.path.abspath(".."))
packages = find_packages(exclude=('tests*', 'docs'))
version = {}
version_file = os.path.join(os.path.dirname(__file__), "../../relezoo", "__init__.py")
with open(version_file) as f:
    exec(f.read(), version)

# -- Project information

project = 'Relezoo'
copyright = '2022, Luis Ferro'
author = 'Luis Ferro'

release = '0.1'
version = version['__version__']

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
