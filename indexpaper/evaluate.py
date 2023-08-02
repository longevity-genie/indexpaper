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


def compute_embeddings_time(model_name: str, input_texts: list[str], cuda: bool = True):
    import torch

    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print(f"trying to use cuda but it is not available!")
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)  # Move the model to the chosen device (GPU if cuda=True and available, otherwise CPU)

    start_time = time.perf_counter()  # Start the timer

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    # Move the batch to the chosen device
    batch_dict = {key: val.to(device) for key, val in batch_dict.items()}

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    end_time = time.perf_counter()  # End the timer

    execution_time = end_time - start_time  # Calculate the execution time
    return embeddings, execution_time

