from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.23'
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
    install_requires=['pycomfort>=0.0.15', 'hybrid-search>=0.0.12', 'more-itertools', 'click', 'python-dotenv', 'tiktoken', 'FlagEmbedding>=1.2.8',
                      'langchain>=0.1.12', "langchain-community>=0.0.28", 'openai', 'Deprecated', 'loguru', '',#'fastembed>=0.2.5',
                      'qdrant-client>=1.8.0', 'sentence_transformers', 'datasets', 'polars>=0.20.14', 'beartype'],
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
