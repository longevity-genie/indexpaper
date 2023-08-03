#!/usr/bin/env python3
import pprint

import click
import loguru
from click import Context
from datasets import load_dataset
from pycomfort.config import LogLevel, LOG_LEVELS, configure_logger
from pycomfort.files import *
import polars as pl
from indexpaper.evaluate import *
from loguru import logger

from indexpaper.evaluate import *
from indexpaper.index import resolve_splitter, EMBEDDINGS, EmbeddingType, index_selected_papers, VectorDatabase


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    # if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

import time


@app.command("preload")
@click.option('--model', default='thenlper/gte-large', help='model to load') #"intfloat/multilingual-e5-large" #"menadsa/S-BioELECTRA"
@click.option('--dataset', default='longevity-genie/tacutu_papers', help='dataset to load')
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def preload(model: str, dataset: str, log_level: str):
    """runs once just to download stuff"""
    configure_logger(log_level)
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    logger.info(f"preloaded {model} model and {dataset} dataset")


@app.command("measure")
@click.option('--model', default='thenlper/gte-large', help='model to load')
@click.option('--dataset', default='longevity-genie/tacutu_papers', help='dataset to load')
@click.option('--embeddings', type=click.Choice(EMBEDDINGS), default=EmbeddingType.HuggingFace.value,
              help='size of the chunk for splitting')
@click.option('--chunk_size', type=click.INT, default=3000, help='size of the chunk for splitting (characters for recursive spliiter and tokens for openai one)')
@click.option("--cuda", type=click.BOOL, default=True, help="should use CUDA")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def measure(model: str, dataset: str, embeddings: str, chunk_size: int, cuda: bool, log_level: str):
    configure_logger(log_level, False)
    df = get_dataset(dataset)
    papers = df.select(pl.col("content_text")).to_series().to_list()
    logger.info(f"computing embedding time for {len(papers)} papers")
    embedding_type = EmbeddingType(embeddings)
    splitter = resolve_splitter(embedding_type, model, chunk_size)
    embeddings, where, timing = index_selected_papers(papers,
                                                      dataset.replace("/", "_"),
                                                      splitter,
                                                      embedding_type,
                                                      database=VectorDatabase.Chroma,
                                                      model=model)
    logger.info(f"the time was {timing} seconds")
    return timing


if __name__ == '__main__':
    app()