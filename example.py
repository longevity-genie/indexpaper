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

@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    # if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

import time


@app.command("preload")
@click.option('--model', default='intfloat/multilingual-e5-large', help='model to load') #"thenlper/gte-large" #"menadsa/S-BioELECTRA"
@click.option('--dataset', default='longevity-genie/tacutu_papers', help='dataset to load')
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def preload(model: str, dataset: str, log_level: str):
    """runs once just to download stuff"""
    configure_logger(log_level)
    _ = load_tokenizer_model_data(model, dataset)
    logger.info(f"preloaded {model} model and {dataset} dataset")


@app.command("measure")
@click.option('--model', default='intfloat/multilingual-e5-large', help='model to load')
@click.option('--dataset', default='longevity-genie/tacutu_papers', help='dataset to load')
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def measure(model: str, dataset: str, log_level: str):
    configure_logger(log_level, False)
    tokenizer, embedding_model, df = load_tokenizer_model_data(model, dataset)
    papers = df.select(pl.col("content_text")).to_series().to_list()
    logger.info(f"computing embedding time for {len(papers)} papers")
    timing = compute_embeddings_time(tokenizer, embedding_model, papers)
    logger.info(f"the time was {timing} seconds")
    return timing


if __name__ == '__main__':
    app()