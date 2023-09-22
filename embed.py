from datasets import load_dataset
import numpy as np
import argparse
from FlagEmbedding import FlagModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str
    )
    parser.add_argument(
        "--embedding_output_path",
        type=str,
        default="embeddings.npy",
        help="Path to save the embeddings, set to None to not save the embeddings"
    )
    parser.add_argument(
        "--dataset_output_path",
        type=str,
        default="embedded_dataset",
        help="Path to save the embedded dataset, set to None to not save the embedded dataset"
    )
    parser.add_argument(
        "--field_to_embed",
        type=str,
        default="text"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-large-en",
        help="Currently only supports BAAI/bge-large-en, BAAI/bge-large-en-v1.5"
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=False,
        help="Whether to normalize the embeddings"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )

    return parser.parse_args()


class BGE:
    def __init__(self, model_name, normalize_embeddings, max_length=512):
        if max_length > 512:
            print("Specified max length is greater than 512, setting to 512")
            max_length = 512
        self.model = FlagModel(model_name, normalize_embeddings=normalize_embeddings, max_length=max_length)

    def encode(self, texts, batch_size=256):
        return self.model.encode_queries(texts, batch_size=batch_size)


def embed_dataset(args):
    dataset = load_dataset(args.dataset_path)["train"]
    model = BGE(args.model_name, args.normalize, args.max_length)
    embeddings = model.encode(list(dataset["text"]), batch_size=args.batch_size)

    if args.embedding_output_path is not None:
        np.save(args.embedding_output_path, embeddings)

    if args.dataset_output_path is not None:
        if "embedding" in dataset.features:
            dataset = dataset.remove_columns("embedding")
        dataset = dataset.map(lambda example, idx: {"embedding": embeddings[idx]}, with_indices=True)
        dataset.save_to_disk(args.dataset_output_path)
    return dataset, embeddings


