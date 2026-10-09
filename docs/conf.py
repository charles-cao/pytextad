import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from pytextad.version import __version__  # noqa: E402

project = "PyTextAD"
author = "Yang Cao"
copyright = "2026, Yang Cao"
version = release = __version__

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "myst_parser"]
autodoc_mock_imports = ["torch", "transformers", "sklearn", "numpy"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "inherited-members": True, "show-inheritance": True}


def _module_docstring_as_literal(app, what, name, obj, options, lines):
    # the module docstrings are plain-text notes (with |x| for absolute values): show them verbatim
    if what == "module" and lines:
        lines[:] = ["::", ""] + ["    " + ln for ln in lines]


def setup(app):
    app.connect("autodoc-process-docstring", _module_docstring_as_literal)
html_theme = "furo"
html_title = f"PyTextAD {version}"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
