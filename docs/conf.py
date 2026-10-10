import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))
# read the version without importing pytextad (its dependencies are not installed on Read the Docs)
with open(os.path.join(os.path.dirname(__file__), "..", "pytextad", "version.py"), encoding="utf-8") as f:
    __version__ = re.search(r'__version__\s*=\s*"([^"]+)"', f.read()).group(1)

project = "PyTextAD"
author = "Yang Cao"
copyright = "2026, Yang Cao"
version = release = __version__

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon",
              "sphinx.ext.viewcode", "myst_parser", "sphinxcontrib.bibtex"]
autodoc_mock_imports = ["torch", "transformers", "sklearn", "numpy", "scipy", "pandas", "matplotlib", "huggingface_hub"]
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autosummary_generate = True
napoleon_use_rtype = False
templates_path = ["_templates"]
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "label"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_extra_path = ["_extra"]          # site-verification file for Google Search Console
html_title = f"PyTextAD {version}"
html_theme_options = {"navigation_depth": 2, "collapse_navigation": False}


def _module_docstring_as_literal(app, what, name, obj, options, lines):
    # module docstrings are plain-text notes (with |x| for absolute values): show them verbatim
    if what == "module" and lines:
        lines[:] = ["::", ""] + ["    " + ln for ln in lines]


def setup(app):
    app.connect("autodoc-process-docstring", _module_docstring_as_literal)
