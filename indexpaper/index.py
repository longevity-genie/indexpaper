#!/usr/bin/env python3
from typing import Dict

import click
from click import Context
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch
from pycomfort.config import load_environment_keys, LOG_LEVELS, LogLevel, configure_logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType

from indexpaper.indexing import index_selected_papers, index_selected_documents, init_qdrant, fast_index_papers
from indexpaper.paperset import Paperset
from indexpaper.resolvers import *
from indexpaper.utils import timing
from opensearchpy import OpenSearch, exceptions

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
@click.option('--folder', type=click.Path(), default=None, help="folder to put chroma indexes to (if we use chroma)")
@click.option('--url', type=click.STRING, default=None, help="alternatively you can provide url, for example http://localhost:6333 for qdrant")
@click.option('--key', type=click.STRING, default=None, help="your api key if you are using cloud vector store")
@click.option('--embeddings', type=click.Choice(EMBEDDINGS), default=EmbeddingType.OpenAI.value, help='size of the chunk for splitting')
@click.option('--chunk_size', type=click.INT, default=3000, help='size of the chunk in tokens for splitting (characters for recursive spliiter and tokens for openai one)')
@click.option("--model", type=click.Path(), default=None, help="path to the model (required for embeddings)")
@click.option('--include_meta', type=click.BOOL, default=True, help="if metadata is included")
@click.option('--database', type=click.Choice(VECTOR_DATABASES, case_sensitive=False), default=VectorDatabase.Chroma.value, help = "which store to take")
@click.option("--device", type=click.Choice(DEVICES), default=Device.cpu.value, help="which device to use, cpu by default, so do not forget to put cuda if you are using NVIDIA")
@click.option('--prefer_grpc', type=click.BOOL, default = None)
@click.option('--rewrite', type=click.BOOL, default=False, help = "Rewrite collection if it is already present")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def index_papers_command(papers: str, collection: str, folder: str, url: str, key: str, embeddings: str, chunk_size: int,  model: Optional[str], include_meta: bool, database: str, device: Optional[str], prefer_grpc: Optional[bool], rewrite: bool, log_level: str) -> Path:
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
@app.command("fast_index_papers")
@click.option('--papers', type=click.Path(exists=True), help="papers folder to index")
@click.option('--collection', default='dataset', help='dataset collection name')
@click.option('--url', type=click.STRING, required=True, help="URL or API key for example http://localhost:6333 for qdrant")
@click.option('--key', type=click.STRING, default=None, help="your api key if you are using cloud vector store")
@click.option("--model", type=click.Path(), default=EmbeddingModels.default.value, help="fast embedding model, BAAI/bge-base-en-v1.5 by default")
@click.option('--prefer_grpc', type=click.BOOL, default=True, help="only needed for qdrant database")
@click.option('--rewrite', type=click.BOOL, default=False, help = "Rewrite collection if it is already present")
@click.option('--paginated', type=click.BOOL, default=True, help = "If it is paginated paper")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def fast_index_papers_command(papers: str, collection: str, url: Optional[str], key: Optional[str], model: str, prefer_grpc: bool, rewrite: bool, paginated: bool, log_level: str) -> Path:
    configure_logger(log_level)
    return fast_index_papers(Path(papers), collection, url, key, model, prefer_grpc, rewrite, paginated)


#@timing
@app.command("hybrid_index")
@click.option('--dataset', type=click.STRING, help="Dataset to index, can be either Path or hugging face dataset")
@click.option('--collection', required=True, help='dataset collection name')
@click.option('--url', type=click.STRING, required=False, help="URL for opensearch, http://localhost:9200 by default for OpenSearch")
@click.option("--model", type=click.Path(), default=EmbeddingModels.default.value, help="fast embedding model, BAAI/bge-base-en-v1.5 by default")
@click.option("--device", type=click.Choice(DEVICES), default=Device.cpu.value, help="which device to use, cpu by default, so do not forget to put cuda if you are using NVIDIA")
@click.option('--start', type=click.INT, default=0, help='When to start slicing the dataset')
@click.option('--content_field', type=click.STRING, default="annotations_paragraph", help = "default dataset content field")
@click.option('--paragraphs', type=click.INT, default=5, help='number of paragraphs to connect together when preprocessing')
@click.option('--slice', type=click.INT, default=100, help='What is the size of the slice')
@click.option('--chunk_size', type=click.INT, default=512, help='What is the size of the chunk')
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def hybrid_index_command(dataset: str, collection: str, url: Optional[str], model: str, device: str, start: int, content_field: str, paragraphs: int, slice: int, chunk_size: int, log_level: str) -> Path:
    logger = configure_logger(log_level)
    load_environment_keys(usecwd=True)
    logger.add("./logs/hybrid_index_{time}.log")
    extras = {
        "dataset": str(dataset),
        "paragraphs_number": str(paragraphs),
        "model": str(model)
    }
    logger.info(f"computing embeddings into collection {collection} for {dataset} with model {model} using slices of {slice} starting from {start} with chunks of {chunk_size} tokens when splitting")
    splitter = HuggingFaceSplitter(model, tokens=chunk_size)
    paper_set = Paperset(dataset, splitter=splitter, content_field=content_field, paragraphs_together=paragraphs)
    embeddings = resolve_embeddings(EmbeddingType.HuggingFaceBGE if "bge" in model else EmbeddingType.HuggingFace, model = model, device = Device(device))
    logger.info(f"computing embeddings into collection {collection} for {dataset} with model {model} using slices of {slice} starting from {start} with chunks of {chunk_size} tokens when splitting")
    url = os.environ.get("OPENSEARCH_URL") if url is None else url
    login = os.getenv("OPENSEARCH_USER", "admin")
    password = os.getenv("OPENSEARCH_PASSWORD", "admin")
    logger.info(f"initializing opensearch at {url}")
    hybrid: OpenSearchHybridSearch = OpenSearchHybridSearch.create(url, collection, embeddings, login=login, password=password)
    #if not hybrid.check_pipeline_exists():
    #    logger.info(f"hybrid search pipeline does not exist, creating it for {url}")
    #    hybrid.create_pipeline(url)
    result = paper_set.index_hybrid_by_slices(slice, hybrid, start, logger=logger, extras = extras)
    return result

import click
from typing import Optional, List

# Assume all the necessary imports and existing command definitions are already in place

@app.command("hybrid_index_multiple")
@click.option('--datasets', multiple=True, type=click.Path(exists=True), help="Datasets to index, can be either Paths or hugging face datasets")
@click.option('--collection', required=True, help='dataset collection name')
@click.option('--url', type=click.STRING, required=False, help="URL for opensearch, http://localhost:9200 by default for OpenSearch")
@click.option("--model", type=click.Path(), default=EmbeddingModels.default.value, help="fast embedding model, BAAI/bge-base-en-v1.5 by default")
@click.option("--device", type=click.Choice(DEVICES), default=Device.cpu.value, help="which device to use, cpu by default, so do not forget to put cuda if you are using NVIDIA")
@click.option('--content_field', type=click.STRING, default="annotations_paragraph", help = "default dataset content field")
@click.option('--paragraphs', type=click.INT, default=5, help='number of paragraphs to connect together when preprocessing')
@click.option('--slice', type=click.INT, default=100, help='What is the size of the slice')
@click.option('--chunk_size', type=click.INT, default=512, help='What is the size of the chunk')
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
@click.pass_context
def hybrid_index_multiple_command(ctx, datasets: List[str], collection: str, url: Optional[str], model: str, device: str, content_field: str, paragraphs: int, slice: int, chunk_size: int, log_level: str):
    logger = configure_logger(log_level)
    load_environment_keys(usecwd=True)
    logger.add("./logs/hybrid_index_multiple_{time}.log")
    for dataset in datasets:
        try:
            ctx.invoke(hybrid_index_command, dataset=dataset, collection=collection, url=url, model=model, device=device, start=0, content_field=content_field, paragraphs=paragraphs, slice=slice, chunk_size=chunk_size, log_level=log_level)
            logger.info(f"Successfully processed dataset: {dataset}")
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
            # Continue to the next dataset without stopping the command execution


@timing
@app.command("fast_index")
@click.option('--dataset', type=click.STRING, help="Dataset to index, can be either Path or hugging face dataset")
@click.option('--collection', default='dataset', help='dataset collection name')
@click.option('--url', type=click.STRING, required=True, help="URL or API key for example http://localhost:6333 for qdrant")
@click.option('--key', type=click.STRING, default=None, help="your api key if you are using cloud vector store")
@click.option("--model", type=click.Path(), default=EmbeddingModels.default.value, help="fast embedding model, BAAI/bge-base-en-v1.5 by default")
@click.option('--prefer_grpc', type=click.BOOL, default=False, help = "only needed for qdrant database")
@click.option('--start', type=click.INT, default=0, help='When to start slicing the dataset')
@click.option('--content_field', type=click.STRING, default="annotations_paragraph", help = "default dataset content field")
@click.option('--paragraphs', type=click.INT, default=5, help='number of paragraphs to connect together when preprocessing')
@click.option('--slice', type=click.INT, default=100, help='What is the size of the slice')
@click.option('--batch_size', type=click.INT, default=50, help="Batch size, 50 by default")
@click.option('--parallel', type=click.INT, default=10, help="How many workers to use")
@click.option('--rewrite', type=click.BOOL, default=False, help = "Rewrite collection if it is already present")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def fast_index_command(dataset: str, collection: str, url: Optional[str], key: Optional[str], model: str, prefer_grpc: bool, start: int, content_field: str, paragraphs: int, slice: int, batch_size: int, parallel: Optional[int], rewrite: bool, log_level: str) -> Path:
    configure_logger(log_level)
    load_environment_keys(usecwd=True)
    assert not (url is None and key is None), "either database url or api_key should be provided!"
    chunk_size: int = 512
    logger.info(f"computing embeddings into collection {collection} for {dataset} with model {model} using slices of {slice} starting from {start} with chunks of {chunk_size} tokens when splitting")
    splitter = HuggingFaceSplitter(model, tokens=chunk_size)
    paper_set = Paperset(dataset, splitter=splitter, content_field=content_field, paragraphs_together=paragraphs)
    api_key = os.getenv("QDRANT_KEY") if key == "QDRANT_KEY" or key == "key" else key
    is_url = "ttp:" in url or "ttps:" in url
    path: Optional[str] = None if is_url else url #actually the user can give either path or url
    url: Optional[str] = url if is_url else None
    logger.info(f"initializing quadrant database at {url}")
    client: QdrantClient = QdrantClient(
        url=url,
        port=6333,
        grpc_port=6334,
        prefer_grpc=is_url if prefer_grpc is None else prefer_grpc,
        api_key=api_key,
        path=path
    )
    client.set_model(model)
    from qdrant_client.http import models as rest
    #client.recreate_collection(collection_name)
    # Just do a single quick embedding to get vector size
    collections = client.get_collections()
    if rewrite or not seq(collections.collections).exists(lambda c: c.name == collection):
        logger.info(f"creating collection {collection}")
        client.recreate_collection(collection_name=collection, vectors_config=client.get_fastembed_vector_params(on_disk=True))
        indexes: dict[str, PayloadSchemaType] = {
            "doi": PayloadSchemaType.TEXT,
            "source": PayloadSchemaType.TEXT,
            "document": PayloadSchemaType.TEXT
        }
        for k, v in indexes.items():
            client.create_payload_index(collection, k, v)
    return paper_set.fast_index_by_slice(n = slice, client=client, collection_name=collection, batch_size=batch_size, parallel=parallel)

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
@click.option('--paragraphs', type=click.INT, default=5, help='number of paragraphs to connect together when preprocessing')
@click.option('--rewrite', type=click.BOOL, default=False, help = "Rewrite collection if it is already present")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def index_dataset_command(dataset: str, collection: str, url: Optional[str], key: Optional[str], embeddings: str, chunk_size: int, model: Optional[str], device: Optional[str], prefer_grpc: bool, slice: int, start: int, content_field: str, paragraphs: int, rewrite: bool, log_level: str) -> Path:
    configure_logger(log_level)
    load_environment_keys(usecwd=True)
    assert not (url is None and key is None), "either database url or api_key should be provided!"
    embedding_type = EmbeddingType(embeddings)
    logger.info(f"computing embeddings into collection {collection} for {dataset} of {embeddings} type with model {model} using slices of {slice} starting from {start} with chunks of {chunk_size} tokens when splitting")
    embedding_function = resolve_embeddings(embedding_type, model = model, device = Device(device))
    splitter = resolve_splitter(embedding_type, model, chunk_size)
    paper_set = Paperset(dataset, splitter=splitter, content_field=content_field, paragraphs_together=paragraphs)
    api_key = os.getenv("QDRANT_KEY") if key == "QDRANT_KEY" or key == "key" else key
    indexes: dict[str, PayloadSchemaType] = {
        "doi": PayloadSchemaType.TEXT,
        "source": PayloadSchemaType.TEXT
    }
    db = init_qdrant(collection, path_or_url=url, embeddings=embedding_function, prefer_grpc=prefer_grpc, always_recreate=rewrite, api_key=api_key, indexes=indexes)
    return paper_set.index_by_slices(slice, db, start = start)


if __name__ == '__main__':
    app()