from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd


def load_or_use_dataset(data):
    """Return a dataset from a path or use the provided one."""
    return load_dataset(data) if isinstance(data, str) else data


def sort_by_keys(dataset, keys):
    """Sort dataset by multiple keys."""
    df = dataset.to_pandas()
    df = df.sort_values(by=keys)
    return Dataset.from_pandas(df)


def format_dataset(path_or_dataset, template_func, filter_key="unique_conversation_id",
                   sort_keys=["dataset_id", "conversation_id", "message_id"], split=None, clustered=False):
    """Format the dataset."""
    dataset = load_or_use_dataset(path_or_dataset)

    if clustered:
        datasets = {cluster_name: cluster_data for cluster_name, cluster_data in dataset.items()}
    else:
        datasets = {"train": dataset.get(split, dataset)}

    formatted_datasets = {}

    for name, data in datasets.items():
        data = sort_by_keys(data, sort_keys)
        formatted = []
        current_filter_key = data[0][filter_key]
        conversation = []

        for message in data:
            if message[filter_key] != current_filter_key:
                formatted += template_func(conversation)
                current_filter_key = message[filter_key]
                conversation = []

            conversation.append(message)

        formatted_datasets[name] = Dataset.from_pandas(pd.DataFrame(data=formatted))

    return DatasetDict(formatted_datasets)


def combine_datasets(datasets, dataset_names):
    def add_dataset_name(example, dataset_name):
        example['dataset_id'] = dataset_name
        return example

    combined = []
    for dataset, name in zip(datasets, dataset_names):
        dataset = dataset.map(lambda example: add_dataset_name(example, name))
        combined += list(dataset)

    return Dataset.from_pandas(pd.DataFrame(data=combined))


def alpaca_template(conversation, max_words=1400, naming_map=None):
    """Format conversation using user and assistant template."""

    conversation = sorted(conversation, key=lambda x: x["message_id"])

    output_text = conversation.pop()['message']
    input_text = ""

    if naming_map is None:
        naming_map = {"instruction": "Instruction",
                      "input": "Input",
                      "output": "Response"}

    for message in conversation:
        text = message['message'].strip()
        if message["message_type"] == "input" and text == "":
            continue
        else:
            input_text += f"### {naming_map[message['message_type']]}:\n{text}\n\n"

    input_text += f"### {naming_map['output']}:\n"

    if len(input_text.split()) > max_words:
        input_text = " ".join(input_text.split(" ")[-max_words:])
        instruction_start = input_text.find(f"### {naming_map['instruction']}:")
        input_text = input_text[instruction_start:]

    return [{"input": input_text, "output": output_text}]


def user_assistant_template(conversation, max_words=1400):
    """Format conversation using user and assistant template."""
    return alpaca_template(conversation, max_words=max_words,
                           naming_map={"instruction": "User",
                                       "output": "Assistant"})