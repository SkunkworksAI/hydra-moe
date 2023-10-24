import argparse
from embed import BGE_Tokenizer
import random
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="configs/classifier_training_config.yaml",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en",
        help="Model to use for embedding, options: 'BAAI/bge-*-en', 'BAAI/bge-*-en-v1.5'"
    )
    parser.add_argument(
        "--normalize_embeddings",
        type=bool,
        default=False,
        help="Whether to normalize the embeddings"
    )

    return parser.parse_args()


class CustomDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len, use_bge):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_bge = use_bge

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])  # convert label to integer
        if self.use_bge:
            # Use BGE to encode the text
            encoding = self.tokenizer.encode([text])
            input_ids = encoding
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        else:
            """Legacy code"""
            # Check if the text is longer than the maximum length
            if len(text.split()) > self.max_len:
                # Calculate the number of tokens to be removed
                num_tokens_to_remove = len(text.split()) - self.max_len
                # Split the text into tokens
                tokens = text.split()
                # Randomly select start and end indices for truncation
                start_index = random.randint(0, num_tokens_to_remove)
                end_index = start_index + self.max_len
                # Truncate the tokens and join them back into a string
                text = " ".join(tokens[start_index:end_index])

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()

        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor([label], dtype=torch.long)
        }


def create_dataset(data, tokenizer, max_len, use_bge):
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return CustomDataset(texts, labels, tokenizer, max_len, use_bge)


def train(args):
    classifier_dataset = load_dataset(args.dataset_path)["train"].to_dict()
    n_labels = len(set(classifier_dataset["label"]))
    train_data, val_data = train_test_split(classifier_dataset, test_size=0.1, random_state=42)

    use_bge = "bge" in args.embedding_model

    if not use_bge:
        raise ValueError("Embedding model must be a BGE model at this time.")

    tokenizer = BGE_Tokenizer(model_name=args.embedding_model, normalize_embeddings=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.embedding_model, num_labels=n_labels)

    train_dataset = create_dataset(train_data, tokenizer, args.max_length, use_bge)
    val_dataset = create_dataset(val_data, tokenizer, args.max_length, use_bge)

    train_config = yaml.safe_load(open(args.train_config_path, "r"))
    training_args = TrainingArguments(**train_config["training_args"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    trainer.save_model("classifier")

    print(trainer.evaluate())


if __name__ == "__main__":
    args = get_args()
    train(args)



