from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader

class BertDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_length=512, batched=True, num_proc=1):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset = self.dataset.map(self.tokenize_function, batched=batched, num_proc=num_proc, remove_columns=["text"])
        self.dataset = self.dataset.map(self.group_texts, batched=batched, num_proc=num_proc, remove_columns=["token_type_ids", "attention_mask"])

    def tokenize_function(self, examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // self.max_seq_length) * self.max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.max_seq_length] for i in range(0, total_length, self.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def mask_example(self, example):
        input_ids = torch.tensor(example['input_ids'])
        special_tokens_mask = torch.tensor(example['special_tokens_mask'])
        label = input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, 0.15)
        special_tokens_mask = special_tokens_mask.to(torch.bool)
        probability_matrix[special_tokens_mask] = 0.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        label[~masked_indices] = -100
        indices_replaced = (torch.bernoulli(torch.full(label.shape, 0.8)).bool() & masked_indices)
        input_ids[indices_replaced] = tokenizer.mask_token_id
        indices_random = (torch.bernoulli(torch.full(label.shape, 0.5)).bool() & masked_indices & ~indices_replaced)
        input_ids[indices_random] = torch.randint(0, tokenizer.vocab_size, input_ids[indices_random].shape)
        return input_ids, label

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input, label = self.mask_example(example)
        return mlm_input, mlm_label

    def __len__(self):
        return len(self.dataset)
