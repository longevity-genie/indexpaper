from typing import Optional

import click
from hybrid_search.opensearch_hybrid_search import OpenSearchHybridSearch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


@click.group(invoke_without_command=True)
@click.pass_context
def app(ctx):
    if ctx.invoked_subcommand is None:
        ctx.invoke(hybrid)

@app.command("hybrid")
@click.option('--url', default='http://localhost:9200', help='URL of the OpenSearch instance')
@click.option('--index', help='Name of the index in OpenSearch')
@click.option('--device', default='cpu', help='Device to run the model on (e.g., cpu, cuda)')
@click.option('--model', default='BAAI/bge-large-en-v1.5', help='Name of the model to use')
@click.option('--query', default='What is ageing?', help='The query to search for')
@click.option('--k', default=10, help='Number of search results to return')
@click.option('--threshold', default=None, help='Threshold to cut out results')
@click.option('--verbose', default=True, help='How much to print')
def hybrid(url: str, index: str, device: str, model: str, query: str, k: int, threshold: Optional[float], verbose: bool):
    print(f"searching in INDEX: {index}, \nQUERY: {query}")
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    docsearch: OpenSearchHybridSearch = OpenSearchHybridSearch.create(url, index, embeddings)
    results = docsearch.hybrid_search(query, k, search_pipeline = "norm-pipeline", threshold=threshold)
    print("Search IDS:")
    for (result, f) in results:
        print(result.metadata["source"], f)
        if verbose:
            print(result)



if __name__ == '__main__':
    app()