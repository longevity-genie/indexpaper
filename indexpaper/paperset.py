#!/usr/bin/env python3
from typing import Any, TypeVar

import langchain
import polars as pl
from beartype import beartype
from datasets import load_dataset
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import VectorStore
from loguru import logger
from pycomfort.files import *

from indexpaper.index import index_selected_papers
from indexpaper.resolvers import *

T = TypeVar('T')

def get_dataset(name: str) -> pl.LazyFrame:
    """
    for example "longevity-genie/moskalev_papers"
    :param name:
    :return: polars Dataframe
    """
    dataset = load_dataset(name)["train"]
    return pl.from_arrow(dataset.data.table).lazy()

def documents_from_dataset_slice(df: pl.DataFrame, splitter: TextSplitter,  paragraphs: str):
    pass


def fold_left_slices(df: pl.LazyFrame, n: int, fold: Callable[[T, pl.DataFrame], T], acc: T, start: int = 0) -> T:
    """
    Function that simulates fold_left on slices of lazy dataframe
    :param df:
    :param n:
    :param fold:
    :param acc:
    :param start:
    :return:
    """
    # Get the slice
    slice_lazy_df = df.slice(start, n)

    # Collect the slice to a DataFrame to check if it has zero rows
    slice_df = slice_lazy_df.collect()
    if slice_df.shape[0] == 0:
        return acc

    # Apply the function to the slice (in place modification)
    upd_acc = fold(acc, slice_df)

    # Recursive call to process the next slice
    return fold_left_slices(df, n, fold, upd_acc, start + n)



def foreach_slice(df: pl.LazyFrame, n: int, fun: Callable[[pl.DataFrame], None], start: int = 0) -> None:
    # Get the slice
    slice_lazy_df = df.slice(start, n)

    # Collect the slice to a DataFrame to check if it has zero rows
    slice_df = slice_lazy_df.collect()
    if slice_df.shape[0] == 0:
        return

    # Apply the function to the slice (in place modification)
    fun(slice_df)

    # Recursive call to process the next slice
    foreach_slice(df, n, fun, start + n)





@beartype
def process_paper_dataset(dataset: Union[pl.LazyFrame, str, Path],
                          collection: str,
                          embedding_type: EmbeddingType,
                          model: str,
                          chunk_size: int = 512,
                          device: Device = Device.cpu)-> (Union[VectorStore, Any, langchain.vectorstores.Chroma], Optional[Union[Path, str]], float):
    if isinstance(dataset, pl.LazyFrame):
        df: pl.LazyFrame = dataset
    elif isinstance(dataset, Path) or ".parquet" in dataset:
        df: pl.LazyFrame = pl.scan_parquet(dataset)
    else:
        df: pl.LazyFrame = get_dataset(dataset)
    papers = df.select(pl.col("content_text")).collect().to_series().to_list()
    splitter: SourceTextSplitter = resolve_splitter(embedding_type, model, chunk_size)
    logger.info(f"computing embedding time for {collection} with {len(papers)} papers")
    db, where, timing = index_selected_papers(papers,
                                              collection,
                                              splitter,
                                              embedding_type,
                                              database=VectorDatabase.Chroma,
                                              model=model, device=device)
    logger.info(f"embeddings of {len(papers)} took {format_time(timing)}")

    return db, where, timing

