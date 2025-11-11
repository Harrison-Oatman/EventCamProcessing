import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "EventCamProcessing"
author = "Joanna Van Liew"

extensions = [
    "sphinx.ext.autodoc",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_markdown_builder",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "furo"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

master_doc = "index"
