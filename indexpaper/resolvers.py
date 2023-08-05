import os
from enum import Enum

from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings, VertexAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from loguru import logger
from pycomfort.files import *

from indexpaper.splitting import OpenAISplitter, SourceTextSplitter, HuggingFaceSplitter

class EmbeddingModels(Enum):
    all_mpnet_base: str = "sentence-transformers/all-mpnet-base-v2"
    bge_base_en: str = "BAAI/bge-base-en" #so far second best at https://huggingface.co/spaces/mteb/leaderboard
    gte_large: str = "thenlper/gte-large"
    gte_base: str = "thenlper/gte-base"
    multilingual_e5_large: str = "intfloat/multilingual-e5-large" #supports many languages and pretty good
    biobert: str = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
    biolord: str = 'FremyCompany/BioLORD-STAMB2-v1' #based on all-mpnet-base-v2 finetined for bio domain
    bioelectra: str = 'menadsa/S-BioELECTRA'
    default: str = bge_base_en


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


def resolve_splitter(embeddings_type: EmbeddingType, model: Optional[Union[Path, str]] = None, chunk_size: Optional[int] = None) -> SourceTextSplitter:
    """
    initializes a splitter based on embeddingtype and additional parameters
    :param embeddings_type:
    :param model:
    :param chunk_size:
    :return:
    """
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
    """
    Initializes embedding instance based on embedding type enum and on additional parameter like model and device
    :param embeddings_type:
    :param model:
    :param device:
    :return:
    """
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

def resolve_embedding_splitter(embeddings_type: EmbeddingType, model: Optional[Union[Path, str]] = None, device: Device = Device.cpu, chunk_size=512):
    return resolve_embeddings(embeddings_type, model, device), resolve_splitter(embeddings_type, model, chunk_size=chunk_size)
