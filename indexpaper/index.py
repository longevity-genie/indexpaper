#!/usr/bin/env python3
import os
from enum import Enum
from pathlib import Path
import time
from typing import Any

import click
from beartype import beartype
from click import Context
from datasets import load_dataset
from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings, VertexAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import langchain
from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma, VectorStore, Qdrant
from loguru import logger
from pycomfort.files import *
from qdrant_client import QdrantClient
import polars as pl

from pycomfort.config import load_environment_keys, LOG_LEVELS, LogLevel, configure_logger

from indexpaper.evaluate import get_dataset
from indexpaper.splitting import OpenAISplitter, SourceTextSplitter, papers_to_documents, HuggingFaceSplitter

class Device(Enum):
    cpu = "cpu"
    cuda = "cuda"
    ipu = "ipu"
    xpu = "xpu"
    mkldnn = "mkldnn"
    opengl = "opengl"
    opencl = "opencl"
    ideep = "ideep"
    hip = "hip"
    ve = "ve"
    fpga = "fpga"
    ort = "ort"
    xla = "xla"
    lazy = "lazy"
    vulkan = "vulkan"
    mps = "mps"
    meta = "meta"
    hpu = "hpu"
    mtia = "mtia"
    privateuseone = "privateuseone"
DEVICES =  [d.value for d in Device]


class VectorDatabase(Enum):
    Chroma = "Chroma"
    Qdrant = "Qdrant"

class EmbeddingType(Enum):
    OpenAI = "openai"
    Llama = "llama"
    VertexAI = "vertexai"
    HuggingFace = "huggingface"

EMBEDDINGS: list[str] = [e.value for e in EmbeddingType]
VECTOR_DATABASES: list[str] = [db.value for db in VectorDatabase]

def get_dataset(name: str) -> pl.LazyFrame:
    """
    for example "longevity-genie/moskalev_papers"
    :param name:
    :return: polars Dataframe
    """
    dataset = load_dataset(name)["train"]
    return pl.from_arrow(dataset.data.table).lazy()


def resolve_splitter(embeddings_type: EmbeddingType, model: Optional[Union[Path, str]] = None, chunk_size: Optional[int] = None) -> SourceTextSplitter:
    if embeddings_type == EmbeddingType.OpenAI:
        if chunk_size is None:
            chunk_size = 3600
        return OpenAISplitter(tokens=chunk_size)
    elif embeddings_type.HuggingFace:
        if chunk_size is None:
            chunk_size = 512
        if model is None:
            logger.error("Model should be specified for Huggingface splitter, using default sentence-transformers/all-mpnet-base-v2 otherwise")
            return HuggingFaceSplitter("sentence-transformers/all-mpnet-base-v2", tokens=chunk_size)
        else:
            return HuggingFaceSplitter(model, tokens=chunk_size)
    else:
        logger.warning(f"{embeddings_type} splitter is not supported, using openai tiktoken based splitter instead")
        return OpenAISplitter(tokens=chunk_size)

def resolve_embeddings(embeddings_type: EmbeddingType, model: Optional[Union[Path, str]] = None, device: Device = Device.cpu) -> Embeddings:
    if embeddings_type == EmbeddingType.OpenAI:
        return OpenAIEmbeddings()
    elif embeddings_type == EmbeddingType.Llama:
        if model is None:
            model = os.getenv("LLAMA_MODEL")
            if model is None:
                logger.error(f"for llama embeddings for {model} model")
            else:
                return LlamaCppEmbeddings(model_path = model)
        return LlamaCppEmbeddings(model_path = str(model))
    elif embeddings_type == EmbeddingType.VertexAI:
        return VertexAIEmbeddings()
    elif embeddings_type == EmbeddingType.HuggingFace:
        if model is None:
            logger.warning(f"for huggingface the model name should be specified")
            return HuggingFaceEmbeddings(model_kwargs={'device': device.value})
        else:
            return HuggingFaceEmbeddings(model_name = str(model), model_kwargs={'device': device.value})
    else:
        logger.warning(f"{embeddings_type.value} is not yet supported by CLI, using default openai embeddings instead")
        return OpenAIEmbeddings()

def db_with_texts(db: VectorStore, texts: list[str],
                 splitter: TextSplitter, id_field: Optional[str] = None, debug: bool = False):
    return db_with_documents(db, texts_to_documents(texts), splitter, id_field, debug)

def db_with_documents(db: VectorStore, documents: list[Document],
                      splitter: TextSplitter,
                      id_field: Optional[str] = None, debug: bool = False):
    """
    Function to add documents to a Chroma database.

    Args:
        db (Chroma): The database to add the documents to.
        documents (list[Document]): The list of documents to add.
        splitter (TextSplitter): The TextSplitter to use for splitting the documents.
        debug (bool): If True, print debug information. Default is False.
        id_field (Optional[str]): If provided, use this field from the document metadata as the ID. Default is None.

    Returns:
        Chroma: The updated database.
    """
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [doc.metadata[id_field] for doc in docs] if id_field is not None else None
    if debug:
        for doc in documents:
            logger.trace(f"ADD TEXT: {doc.page_content}")
            logger.trace(f"ADD METADATA {doc.metadata}")
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return db

def init_qdrant(collection_name: str, path_or_url: str,  embedding_function: Optional[Embeddings], api_key: Optional[str] = None, distance_func: str = "Cosine", prefer_grpc: bool = False):
    is_url = "ttp:" in path_or_url or "ttps:" in path_or_url
    path: Optional[str] = None if is_url else path_or_url
    url: Optional[str] = path_or_url if is_url else None
    logger.info(f"initializing quadrant database at {path_or_url}")
    client: QdrantClient = QdrantClient(
        url=url,
        port=6333,
        grpc_port=6334,
        prefer_grpc=is_url if prefer_grpc is None else prefer_grpc,
        api_key=api_key,
        path=path
    )
    from qdrant_client.http import models as rest
    #client.recreate_collection(collection_name)
    # Just do a single quick embedding to get vector size
    partial_embeddings = embedding_function.embed_documents("probe")
    vector_size = len(partial_embeddings[0])
    distance_func = distance_func.upper()
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance[distance_func],
        )
    )
    return Qdrant(client, collection_name=collection_name, embeddings=embedding_function)


def write_remote_db(url: str,
                    collection_name: str,
                    documents: list[Document],
                    splitter: TextSplitter,
                    id_field: Optional[str] = None,
                    embeddings: Optional[Embeddings] = None,
                    database: VectorDatabase = VectorDatabase.Qdrant,
                    key: Optional[str] = None, prefer_grpc: Optional[bool] = False) -> (Union[VectorStore, Any, langchain.vectorstores.Chroma], Optional[Union[Path, str]], float):
    if database == VectorDatabase.Qdrant:
        logger.info(f"writing a collection {collection_name} of {len(documents)} documents to quadrant db at {url}")
        start_time = time.perf_counter()
        api_key = os.getenv("QDRANT_KEY") if key == "QDRANT_KEY" or key == "key" else key
        db = init_qdrant(collection_name, path_or_url=url, embedding_function=embeddings, api_key=api_key, prefer_grpc=prefer_grpc)
        db_updated = db_with_documents(db, documents, splitter,  id_field)
        end_time = time.perf_counter()
        execution_time: float = end_time - start_time
        return db_updated, url, execution_time
    else:
        raise Exception(f"Remote Chroma is not yet supported by this script!")
    pass

@beartype
def make_local_db(collection_name: str,
                   documents: list[Document],
                   splitter: TextSplitter,
                   embeddings: Optional[Embeddings] = None,
                   database: VectorDatabase = VectorDatabase.Chroma,
                   persist_directory: Optional[Path] = None,
                   id_field: Optional[str] = None,
                   prefer_grpc: Optional[bool] = False
                   ) -> (Union[VectorStore, Any, langchain.vectorstores.Chroma], Optional[Union[Path, str]], float):
    """
    :param collection_name:
    :param documents:
    :param splitter:
    :param embeddings:
    :param database:
    :param persist_directory:
    :param id_field:
    :param prefer_grpc:
    :return:
    """
    start_time = time.perf_counter()

    # If no embeddings were provided, default to OpenAIEmbeddings
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    # Create the directory where the database will be saved, if it doesn't already exist
    if persist_directory is not None:
        where = persist_directory / collection_name
        where.mkdir(exist_ok=True, parents=True)
        where_str = str(where)
    else:
        where = None
        where_str = None

    # Create a Chroma database with the specified collection name and embeddings, and save it in the specified directory
    if database == VectorDatabase.Qdrant:
        db = init_qdrant(collection_name, where_str, embedding_function=embeddings,  prefer_grpc = prefer_grpc)
    else:
        db = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings)
    db_updated = db_with_documents(db, documents, splitter,  id_field)

    # Persist the changes to the database
    if persist_directory is not None:
        db_updated.persist()
    end_time = time.perf_counter()
    execution_time: float = end_time - start_time

    # Return the directory where the database was saved
    return db_updated, where, execution_time


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    # if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass


@beartype
def texts_to_documents(texts: list[str]) -> list[Document]:
    return [Document(
        page_content=text,
        metadata={"text": text}
    ) for text in texts]

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

@beartype()
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

@beartype
def index_selected_papers(folder_or_texts: Union[Path, list[str]],
                          collection: str,
                          splitter: SourceTextSplitter,
                          embedding_type: EmbeddingType,
                          include_meta: bool = True,
                          folder: Optional[Union[Path,str]] = None,
                          url: Optional[str] = None,
                          key: Optional[str] = None,
                          database: VectorDatabase = VectorDatabase.Chroma.value,
                          model: Optional[Union[Path, str]] = None,
                          prefer_grpc: Optional[bool] = None,
                          device: Device = Device.cpu
                          ) -> (Union[VectorStore, Any, langchain.vectorstores.Chroma], Optional[Union[Path, str]], float):
    openai_key = load_environment_keys() #for openai key
    embeddings_function = resolve_embeddings(embedding_type, model, device)
    logger.info(f"embeddings are {embedding_type}")
    documents = papers_to_documents(folder_or_texts, include_meta=include_meta) if isinstance(folder_or_texts, Path) else texts_to_documents(folder_or_texts)
    if url is not None or key is not None:
        return write_remote_db(url, collection, documents, splitter, embeddings=embeddings_function, database=database, key=key, prefer_grpc = prefer_grpc)
    else:
        if folder is None:
            logger.warning(f"neither url not folder are set, trying in memory")
            return make_local_db(collection, documents, splitter, embeddings_function, prefer_grpc = prefer_grpc, database=database)
        else:
            index = Path(folder) if isinstance(folder, str) else folder
            index.mkdir(exist_ok=True)
            where = index / f"{embedding_type.value}_{splitter.chunk_size}_chunk"
            where.mkdir(exist_ok=True, parents=True)
            logger.info(f"writing index of papers to {where}")
            return make_local_db(collection, documents, splitter, embeddings_function, persist_directory=where,  prefer_grpc = prefer_grpc, database=database)


@app.command("index_papers")
@click.option('--papers', type=click.Path(exists=True), help="papers folder to index")
@click.option('--collection', default='papers', help='papers collection name')
@click.option('--folder', type=click.Path(), default=None, help="folder to put chroma indexes to")
@click.option('--url', type=click.STRING, default=None, help="alternatively you can provide url, for example http://localhost:6333 for qdrant")
@click.option('--key', type=click.STRING, default=None, help="your api key if you are using cloud vector store")
@click.option('--embeddings', type=click.Choice(EMBEDDINGS), default=EmbeddingType.OpenAI.value,
              help='size of the chunk for splitting')
@click.option('--chunk_size', type=click.INT, default=3000, help='size of the chunk for splitting (characters for recursive spliiter and tokens for openai one)')
@click.option("--model", type=click.Path(), default=None, help="path to the model (required for embeddings)")
@click.option('--include_meta', type=click.BOOL, default=True, help="if metadata is included")
@click.option('--database', type=click.Choice(VECTOR_DATABASES, case_sensitive=False), default=VectorDatabase.Chroma.value, help = "which store to take")
@click.option("--device", type=click.Choice(DEVICES), default=Device.cpu.value, help="which device to use, cpu by default, so do not forget to put cuda if you are using NVIDIA")
@click.option('--prefer_grpc', type=click.BOOL, default = None)
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def index_papers_command(papers: str, collection: str, folder: str, url: str, key: str, embeddings: str, chunk_size: int, model: Optional[str], include_meta: bool, database: str, device: Optional[str], prefer_grpc: Optional[bool], log_level: str) -> Path:
    configure_logger(log_level)
    load_environment_keys(usecwd=True)
    papers_folder = Path(papers)
    assert not (folder is None and url is None and key is None), "either database folder or database url or api_key should be provided!"
    embedding_type = EmbeddingType(embeddings)
    splitter = resolve_splitter(embedding_type, model, chunk_size)
    return index_selected_papers(papers_folder, collection, splitter,  embedding_type, include_meta, folder, url,
                                 database=VectorDatabase[database],
                                 key=key,
                                 model=model,
                                 prefer_grpc=prefer_grpc,
                                 device=Device(device)
                                 )


if __name__ == '__main__':
    app()