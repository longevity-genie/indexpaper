import time
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModel
import polars as pl
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def get_dataset(name: str) -> pl.DataFrame:
    """
    for example "longevity-genie/moskalev_papers"
    :param name:
    :return: polars Dataframe
    """
    dataset = load_dataset(name)["train"]
    return pl.from_arrow(dataset.data.table)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_tokenizer_model_data(model: str, dataset: str):
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    df = get_dataset(dataset)
    return tokenizer, model, df


def compute_embeddings_time(tokenizer: PreTrainedTokenizerBase, model, input_texts: list[str]):
    start_time = time.perf_counter()  # Start the timer

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    end_time = time.perf_counter()  # End the timer

    execution_time = end_time - start_time  # Calculate the execution time

    return execution_time
