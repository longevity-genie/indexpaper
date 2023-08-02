#!/usr/bin/env python3
import pprint

import click
from click import Context
from datasets import load_dataset
from pycomfort.files import *
import polars as pl


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    # if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

def download(base = Path(".")):
    dataset = load_dataset("longevity-genie/moskalev_papers")["train"]
    df = pl.from_arrow(dataset.data.table)
    return df

@app.command("download")
def download_command():
    df = download()
    pprint.pprint(df)

if __name__ == '__main__':
    app()