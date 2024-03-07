# indexpaper

The project devoted to indexing papers in vector databases

It was originally part of getpaper but now has not dependencies on it

We provide features to index the papers as well as semantic-scholar paper datasets with openai, huggingface or llama embeddings and save them either chromadb or qdrant vector store.

For openai embeddings to work you have to create .env file and specify your openai key there, see .env.template as example

# getting started

Install the library with:
```bash
pip install indexpaper
```

On linux systems you sometimes need to check that build--essential are installed:
```bash
sudo apt install build-essential
```
It is also recommended to use micromamba, conda, anaconda or other environments to avoid bloating system python with too many dependencies.
Assuming you installed [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html), you will have to create an environment and enable the library locally
```
micromamba create -f environment.yaml
micromabma activate indexpaper
pip install -e .
```
The last command is optional. With conda/anaconda the commands will look the same but with another name of executable.

## Running local Qdrant

We provide docker-compose configuration to run local qdrant (you can also use qdrant cloud instead).
To run local qdrant install docker compose (sometimes needs sudo) and run:
```bash
cd services
docker compose up
```
Then you should be able to see  http://localhost:6333/dashboard for qdrant dashboard and http://0.0.0.0:5601 for OpenSearch dashboard

# Additional requirements

index.py has local dependencies on other modules, for this reason if you are running it inside indexpaper project folder consider having it installed locally:
```bash
pip install -e .
```

# indexing a dataset

To index a dataset you can use either index.py dataset subcommand or you look how to do it in code in papers.ipynb example notebook
For example, if we want to index "longevity-genie/tacutu_papers" huggingface dataset using "michiyasunaga/BioLinkBERT-large" hugging face embedding with "cuda" as device and with 10 papers in a slice.
And we want to write it to the local version of qdrant located at http://localhost:6333 (see services for docker-compose file):
```bash
python indexpaper/index.py dataset --collection bge_large_v1.5_512_tacutu_papers_paragraphs_10 --dataset "longevity-genie/tacutu_papers" --url http://localhost:6333 --model BAAI/bge-large-en-v1.5 --slice 10 --chunk_size 500 --device cuda
```

Another example. If we want to index "longevity-genie/moskalev_papers" huggingface dataset using "michiyasunaga/BioLinkBERT-large" hugging face embedding with "gpu" as device and with 10 papers in a slice.
And we want to use our Qdrant cloud key (fill in QDRANT_KEY or put it to environment variable)

Another example. Robi Tacutu papers with cpu using QDRANT_KEY, cluster url (put yours) and michiyasunaga/BioLinkBERT-large embeddings model:
```
python indexpaper/index.py dataset --url https://62d4a96e-2b91-4ab8-a4dd-a91e626d874a.europe-west3-0.gcp.cloud.qdrant.io:6333 --collection biolinkbert_large_512_tacutu_papers --embeddings huggingface --dataset "longevity-genie/tacutu_papers" --key QDRANT_KEY --model michiyasunaga/BioLinkBERT-large --slice 500 --chunk_size 512 --device cpu
```
If  you do not specify  embeddings, slice and chunk, then BGE-large-en with chunk-size 512 and slice of 100 is used by default:
```
python indexpaper/index.py dataset --collection bge_large_v1.5_512_moskalev_papers_paragraphs_10 --dataset "longevity-genie/moskalev_papers" --url https://62d4a96e-2b91-4ab8-a4dd-a91e626d874a.europe-west3-0.gcp.cloud.qdrant.io:6333 --key QDRANT_KEY
```
If you want to recreate the collection from scratch you can also add --rewrite true


# Fast indexing

We also experimentally support fast indexing that has similar parameters, for example:

Robi Tacutu papers QDRANT_KEY, cluster url (put yours) and  embeddings model:
```
python indexpaper/index.py fast_index --url https://62d4a96e-2b91-4ab8-a4dd-a91e626d874a.europe-west3-0.gcp.cloud.qdrant.io:6333 --collection bge_base_en_v1.5_tacutu_papers_5 --dataset "longevity-genie/tacutu_papers" --key QDRANT_KEY --paragraphs 5 --model BAAI/bge-base-en-v1.5 --slice 100 --batch_size 50 --parallel 10
```

# OpenSearch hybrid indexing

For example indexing with bge
```
python indexpaper/index.py hybrid_index --collection tacutu_papers_bge_base_en_v1.5 --model "BAAI/bge-base-en-v1.5" --dataset "longevity-genie/tacutu_papers"
```
If you want to index it with gpu and at different host (for example pic) use:
```
python indexpaper/index.py hybrid_index --collection tacutu_papers_bge_base_en_v1.5 --model "BAAI/bge-base-en-v1.5" --url "https://pic:9200" --dataset "longevity-genie/tacutu_papers" --device cuda
```

You can also make a test search:
```
python indexpaper/search.py hybrid --index "tacutu_papers_bge_base_en_v1.5" --model "BAAI/bge-base-en-v1.5" --query "mitochondrial GC content and longevity" --verbose True --k 3 --verbose True
```

Same for specter model:
```
python indexpaper/index.py hybrid_index --collection specter2_tacutu_papers --model "allenai/specter2_base" --dataset "longevity-genie/tacutu_papers"
```

You can also make a test search:
```
python indexpaper/search.py hybrid --index "specter2_tacutu_papers" --model "allenai/specter2_base" --query "mitochondrial GC content and longevity" --k 3 --verbose true
```

# Indexing papers

For example if you have your papers inside data/output/test/papers folder, and you want to make an index at data/output/test/index you can do it by:
```bash
indexpaper/index.py index_papers --papers data/output/test/papers --folder data/output/test/index --collection mypapers --chunk_size 6000
```

It is possible to use both Chroma and Qdrant. To use qdrant we provide docker-compose file to set it up:
```bash
cd services
docker compose -f docker-compose.yaml up
```
then you can run the indexing of the paper with Qdrant:
```
indexpaper/index.py index_papers --papers data/output/test/papers --url http://localhost:6333 --collection mypapers --chunk_size 6000 --database Qdrant
```
You can also take a look if things were added to the collection with qdrant web UI by checking http://localhost:6333/dashboard

### Checking SemanticScholar dataset

We provide some convenience methods to also run semantic-scholar datasets API.
For example, if you want to get s2orc dataset you can run:
```bash
python indexpaper/check_scholar.py --key <your_semantic_schoalr_key> https://api.semanticscholar.org/datasets/v1/release/latest s2orc --output s2orc.json
```

If you need just the files you can do:
```bash
python indexpaper/check_scholar.py --key <your_semantic_schoalr_key> https://api.semanticscholar.org/datasets/v1/release/latest s2orc --output s2orc.json
```

### Indexing with Llama-2 embeddings ###
You can also use llama-2 embeddings if you install llama-cpp-python and pass a path to the model, for example for https://huggingface.co/TheBloke/Llama-2-13B-GGML model:
```
indexpaper/index.py index_papers --papers data/output/test/papers --url http://localhost:6333 --collection papers_llama2_2000 --chunk_size 2000 --database Qdrant --embeddings llama --model /home/antonkulaga/sources/indexpaper/data/models/llama-2-13b-chat.ggmlv3.q2_K.bin
```
Instead of explicitly pathing the model path you can also include the path to LLAMA_MODEL to the .env file as:
```
LLAMA_MODEL="/home/antonkulaga/sources/indexpaper/data/models/llama-2-13b-chat.ggmlv3.q2_K.bin"
```
Note: if you want to use Qdrant cloud you do not need docker-compose, but you need to provide a key and look at qdrant cloud setting for the url to give.
```
indexpaper/index.py index_papers --papers data/output/test/papers --url https://5bea7502-97d4-4876-98af-0cdf8af4bd18.us-east-1-0.aws.cloud.qdrant.io:6333 --key put_your_key_here --collection mypapers --chunk_size 6000 --database Qdrant
```
Note: there are temporal issues with embeddings for llama.