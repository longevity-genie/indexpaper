#!/usr/bin/env python3
from typing import Dict

import click
from click import Context
from pycomfort.config import load_environment_keys, LOG_LEVELS, LogLevel, configure_logger
from qdrant_client.http.models import PayloadSchemaType

from indexpaper.indexing import index_selected_papers, index_selected_documents, init_qdrant
from indexpaper.paperset import Paperset
from indexpaper.resolvers import *
from indexpaper.utils import timing

@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    # if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass


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
@click.option('--rewrite', type=click.BOOL, default=False, help = "Rewrite collection if it is already present")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def index_papers_command(papers: str, collection: str, folder: str, url: str, key: str, embeddings: str, chunk_size: int, model: Optional[str], include_meta: bool, database: str, device: Optional[str], prefer_grpc: Optional[bool], rewrite: bool, log_level: str) -> Path:
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
                                 device=Device(device),
                                 always_recreate=rewrite
                                 )
@timing
@app.command("dataset")
@click.option('--dataset', type=click.STRING, help="Dataset to index, can be either Path or hugging face dataset")
@click.option('--collection', default='dataset', help='dataset collection name')
@click.option('--url', type=click.STRING, required=True, help="URL or API key for example http://localhost:6333 for qdrant")
@click.option('--key', type=click.STRING, default=None, help="your api key if you are using cloud vector store")
@click.option('--embeddings', type=click.Choice(EMBEDDINGS), default=EmbeddingType.HuggingFaceBGE.value,
              help='embeddings type, huggingface by default')
@click.option('--chunk_size', type=click.INT, default=512, help='size of the chunk for splitting (characters for recursive spliter and tokens for openai one)')
@click.option("--model", type=click.Path(), default=EmbeddingModels.default.value, help="path to the model (required for embeddings)")
@click.option("--device", type=click.Choice(DEVICES), default=Device.cpu.value, help="which device to use, cpu by default, so do not forget to put cuda if you are using NVIDIA")
@click.option('--prefer_grpc', type=click.BOOL, default=False, help = "only needed for qdrant database")
@click.option('--slice', type=click.INT, default=100, help='What is the size of the slice')
@click.option('--start', type=click.INT, default=0, help='When to start slicing the dataset')
@click.option('--content_field', type=click.STRING, default="annotations_paragraph", help = "default dataset content field")
@click.option('--rewrite', type=click.BOOL, default=False, help = "Rewrite collection if it is already present")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def index_dataset_command(dataset: str, collection: str, url: Optional[str], key: Optional[str], embeddings: str, chunk_size: int, model: Optional[str], device: Optional[str], prefer_grpc: bool, slice: int, start: int, content_field: str, rewrite: bool, log_level: str) -> Path:
    configure_logger(log_level)
    load_environment_keys(usecwd=True)
    assert not (url is None and key is None), "either database url or api_key should be provided!"
    embedding_type = EmbeddingType(embeddings)
    logger.info(f"computing embeddings into collection {collection} for {dataset} of {embeddings} type with model {model} using slices of {slice} starting from {start} with chunks of {chunk_size} tokens when splitting")
    embedding_function = resolve_embeddings(embedding_type, model = model, device = Device(device))
    splitter = resolve_splitter(embedding_type, model, chunk_size)
    paper_set = Paperset(dataset, splitter=splitter, content_field=content_field)
    api_key = os.getenv("QDRANT_KEY") if key == "QDRANT_KEY" or key == "key" else key
    indexes: dict[str, PayloadSchemaType] = {
        "doi": PayloadSchemaType.TEXT,
        "source": PayloadSchemaType.TEXT
    }
    db = init_qdrant(collection, path_or_url=url, embeddings=embedding_function, prefer_grpc=prefer_grpc, always_recreate=rewrite, api_key=api_key, indexes=indexes)
    return paper_set.index_by_slices(slice, db, start = start)


if __name__ == '__main__':
    app()