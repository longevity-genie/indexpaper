import time

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


#def average_pool(hidden_state, attention_mask):
#    # Assuming you want to perform mean pooling over the non-padding part of the sequence
#    sum_hidden_state = torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=1)
#    count_hidden_state = torch.sum(attention_mask, dim=1).unsqueeze(-1)
#    return sum_hidden_state / count_hidden_state


def compute_embeddings_time(model_name: str, input_texts: list[str], cuda: bool = True):
    import torch

    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print(f"trying to use cuda but it is not available!")
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    total_execution_time = 0
    embeddings_list = []

    for text in input_texts:
        start_time = time.perf_counter()

        # Tokenize the input text
        batch_dict = tokenizer([text], max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {key: val.to(device) for key, val in batch_dict.items()}

        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"Execution time for text: {execution_time:.4f} seconds")
        total_execution_time += execution_time
        embeddings_list.append(embeddings)

    print(f"Total execution time: {total_execution_time:.4f} seconds")
    return torch.cat(embeddings_list, dim=0), total_execution_time

