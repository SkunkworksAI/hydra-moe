from datasets import Dataset, load_dataset
import pandas as pd


def add_unique_conversation_id(example, dataset_name):
    example["unique_conversation_id"] = f"{dataset_name}_{example['conversation_id']}"
    return example


def sort_Dataset_by_multiple_keys(dataset, keys):
    df = dataset.to_pandas()
    df = df.sort_values(by=keys)
    return Dataset.from_pandas(df)


def format(path_or_dataset, template_func, filter_key="unique_conversation_id",
           sort_keys=["dataset_id", "conversation_id", "message_id"], split="train"):
    dataset = load_dataset(path_or_dataset) if isinstance(path_or_dataset, str) else path_or_dataset
    data = dataset[split] if split else dataset
    data = sort_Dataset_by_multiple_keys(data, sort_keys)

    formatted = []
    dataset_has_instructions = has_instructions(data)

    current_filter_key = data[0][filter_key]
    conversation = []

    for message in data:
        if message[filter_key] != current_filter_key:
            formatted += template_func(conversation, dataset_has_instructions)
            current_filter_key = message[filter_key]
            conversation = []
        conversation += [message]

    return Dataset.from_pandas(pd.DataFrame(data=formatted))


def combine_datasets(datasets, dataset_names, add_unique_id=False):
    def add_dataset_name(example, dataset_name):
        example["dataset_id"] = dataset_name
        if add_unique_id:
            example = add_unique_conversation_id(example, dataset_name)
        return example

    new_datasets = []
    # add column relating to index of dataset
    for idx, dataset in enumerate(datasets):
        if "train" in dataset.keys():
            dataset = dataset["train"]
        short_name = "_".join(dataset_names[idx].split("/")[1].split("_")[:-1])
        new_datasets.append(dataset.map(
            lambda example: add_dataset_name(example, short_name)
        ))

    combined = list(new_datasets[0])
    for dataset in new_datasets[1:]:
        combined += list(dataset)
    return Dataset.from_pandas(pd.DataFrame(data=combined))


def has_instructions(dataset):
    return "instruction" in set(dataset["message_type"])


def cluster_template(
        conversation, dataset_has_instructions="not used for this template"
):
    conversation = sorted(conversation, key=lambda x: x["message_id"])
    combined_text = ""
    for message in conversation:
        combined_text += f"{message['message']}\n"
    return [
        {"text": combined_text, "conversation_id": conversation[0]["conversation_id"],
         "dataset_id": conversation[0]["dataset_id"],
         "unique_conversation_id": conversation[0]["unique_conversation_id"]}
    ]


def alpaca_template(conversation, dataset_has_instructions, max_words=1400):
    # sort by message_id
    conversation = sorted(conversation, key=lambda x: x["message_id"])
    output = f"{conversation.pop()['message']}"
    input = ""

    for message in conversation:
        if message["message_type"] == "instruction":
            input += f"### Instruction:\n{message['message']}\n\n"
        elif message["message_type"] == "output":
            input += f"### Response:\n{message['message']}\n"
        elif (
                message["message_type"] == "input"
                and dataset_has_instructions
                and message["message"] != ""
        ):
            input += f"### Input:\n{message['message']}\n\n"
        elif not dataset_has_instructions:
            input += f"### Instruction:\n{message['message']}\n\n"
    input += "### Response:\n"

    if len(input.split(" ")) > max_words:
        input = " ".join(input.split()[-max_words:])

    instruction_start = input.find("### Instruction:")
    input = input[instruction_start:]
    return [{"input": input, "output": output}]
