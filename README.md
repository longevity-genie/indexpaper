# indexpaper
The project devoted to indexing papers in vector databases

It was originally part of getpaper but now has not dependencies on it

We provide features to index the papers with openai or llama embeddings and save them in chromadb vector store.
For openai embeddings to work you have to create .env file and specify your openai key there, see .env.template as example

For example if you have your papers inside data/output/test/papers folder, and you want to make a ChromaDB index at data/output/test/index you can do it by:
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

# Examples

You can run examples.py to see usage examples and also to evaluate embeddings.

For example if you want to evaluate how fast embeddings compute on Robi Tacutu papers you can run:
```
python example.py preload
```
to download dataset and model. And then:
```bash
python example.py evaluate --model intfloat/multilingual-e5-large --dataset longevity-genie/tacutu_papers
```
To measure time
```bash
python example.py measure --model intfloat/multilingual-e5-large --dataset longevity-genie/tacutu_papers
```

# indexing a dataset

To index a dataset you can use either index.py dataset subcomment or you look how to do it in code in papers.ipynb example notebook
For example, if we want to index "longevity-genie/tacutu_papers" huggingface dataset using "michiyasunaga/BioLinkBERT-large" hugging face embedding with "gpu" as device and with 10 papers in a slice.
And we want to write it to the local version of qdrant located at http://localhost:6333 (see services for docker-compose file):
```bash
python indexpaper/index.py dataset --collection biolinkbert_512_tacutu_papers --dataset "longevity-genie/tacutu_papers" --url http://localhost:6333 --model michiyasunaga/BioLinkBERT-large --slice 10 --chunk_size 512 --device gpu
```

Another example. If we want to index "longevity-genie/moskalev_papers" huggingface dataset using "michiyasunaga/BioLinkBERT-large" hugging face embedding with "gpu" as device and with 10 papers in a slice.
And we want to use our Qdrant cloud key (fill in QDRANT_KEY or put it to environment variable)
```bash
python indexpaper/index.py dataset --collection biolinkbert_512_moskalev_papers --dataset "longevity-genie/moskalev_papers" --key QDRANT_KEY --model michiyasunaga/BioLinkBERT-large --slice 10 --chunk_size 512 --device gpu
```
Another example. Robi Tacutu papers with cpu using QDRANT_KEY and biolord embeddings model:
```
python indexpaper/index.py dataset --collection biolord_512_tacutu_papers --dataset "longevity-genie/tacutu_papers" --key QDRANT_KEY --model FremyCompany/BioLORD-STAMB2-v1 --slice 10 --chunk_size 512 --device cpu
```

# Runnning local Qdrant

We provide docker-compose configuration to run local qdrant (you can also use qdrant cloud instead).
To run local qdrant install dockerr compose and run:
```bash
cd services
docker compose up
```
Then you should be able to see  http://localhost:6333/dashboard

# Additional requirements

index.py has local dependencies on other modules, for this reason if you are running it inside indexpaper project folder consider having it installed locally:
```bash
pip install -e .
```
