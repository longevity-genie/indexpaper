from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.11'
DESCRIPTION = 'indexpaper - library to index papers with vector databases'
LONG_DESCRIPTION = 'indexpaper - library to index papers with vector databases'

# Setting up
setup(
    name="indexpaper",
    version=VERSION,
    author="antonkulaga (Anton Kulaga)",
    author_email="<antonkulaga@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pyfunctional', 'pycomfort', 'more-itertools', 'click', 'python-dotenv', 'tiktoken',
                      'langchain', 'openai', 'Deprecated', 'loguru',
                      'qdrant-client', 'chromadb', 'sentence_transformers', 'datasets', 'polars', 'beartype'],
    keywords=['python', 'utils', 'files', 'papers', 'download', 'index', 'vector databases'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points={
     "console_scripts": [
         "index=indexpaper.index:app"
     ]
    }
)
