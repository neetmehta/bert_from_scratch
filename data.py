from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader

class BertMLMDataset(Dataset):
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

class BertNSPDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer: BertTokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Precompute sentence pairs
        self.sentence_pairs = self._prepare_sentence_pairs()

    def _split_sentences(self, text):
        # Use tokenizer's built-in sentence splitter if available
        # Otherwise use a simple heuristic
        return [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 30]

    def _prepare_sentence_pairs(self):
        pairs = []
        for sample in self.dataset:
            sentences = self._split_sentences(sample["content"])
            if len(sentences) < 2:
                continue

            for i in range(len(sentences) - 1):
                is_next = random.random() > 0.5

                sent_a = sentences[i]
                if is_next:
                    sent_b = sentences[i + 1]
                    label = 0  # IsNext
                else:
                    # Sample random sentence B from another document
                    rand_doc = random.choice(self.dataset)
                    rand_sents = self._split_sentences(rand_doc["content"])
                    if rand_sents:
                        sent_b = random.choice(rand_sents)
                        label = 1  # NotNext
                    else:
                        continue  # skip if empty

                pairs.append((sent_a, sent_b, label))
        return pairs

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sent_a, sent_b, label = self.sentence_pairs[idx]

        encoding = self.tokenizer(
            sent_a,
            sent_b,
            truncation=True,
            return_tensors="pt"
        )

        return encoding["input_ids"].squeeze(0), torch.tensor(label, dtype=torch.long)
