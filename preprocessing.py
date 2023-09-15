from datasets import Dataset, load_dataset
import pandas as pd


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


def list_dict_template(
    conversation, dataset_has_instructions="not used for this template"
):
    # sort by message_id
    conversation = sorted(conversation, key=lambda x: x["message_id"])
    formatted = list()
    turn = dict()
    for message in conversation:
        if message["message_type"] == "instruction":
            turn["instruction"] = message["message"]
        elif message["message_type"] == "input":
            turn["input"] = message["message"]
        else:
            turn["response"] = message["message"]
            formatted.append(turn)
            turn = dict()
    return [
        {
            "conversations": formatted,
            "conversation_id": conversation[0]["conversation_id"],
        }
    ]


def cluster_template(
    conversation, dataset_has_instructions="not used for this template"
):
    conversation = sorted(conversation, key=lambda x: x["message_id"])
    combined_text = ""
    for message in conversation:
        combined_text += f"{message['message']}\n"
    return [
        {"text": combined_text, "conversation_id": conversation[0]["conversation_id"]}
    ]


def format(path, template_func, filter_key="conversation_id", split="train", config = None):
    if config is None:
        dataset = load_dataset(path)
    else:
        dataset = load_dataset(path, config)
    formatted = []
    data = dataset[split] if split else dataset
    data = data.sort(filter_key)

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


def combine_datasets(datasets, dataset_names):
    def add_dataset_name(example, dataset_name):
        example["dataset_id"] = dataset_name
        return example

    # add column relating to index of dataset
    for idx, dataset in enumerate(datasets):
        short_name = "_".join(dataset_names[idx].split("/")[1].split("_")[:-1])
        datasets[idx] = dataset.map(
            lambda example: add_dataset_name(example, short_name)
        )

    combined = list(datasets[0])
    for dataset in datasets[1:]:
        combined += list(dataset)
    return Dataset.from_pandas(pd.DataFrame(data=combined))


def has_instructions(dataset):
    return "instruction" in set(dataset["message_type"])


if __name__ == "__main__":
    # list_of_datasets = ['HydraLM/GPTeacher-General-Instruct_standardized', 'HydraLM/airoboros-gpt4-1.4_standardized',
    #                     'HydraLM/GPT4-LLM-Cleaned_standardized',
    #                     'HydraLM/WizardLM_evol_instruct_V2_196k_standardized',
    #                     'HydraLM/CodeAlpaca-20k_standardized',
    #                     'HydraLM/unnatural-instructions_standardized']
    # list_of_datasets += [f'HydraLM/{subject}_dataset_standardized' for subject in ['biology', 'chemistry', 'math', 'physics']]

    # list_dict generation
    # processed_datasets = [format(dataset, list_dict_template) for dataset in list_of_datasets]
    # for idx, dataset_name in enumerate(list_of_datasets):
    #     new_name = f'{dataset_name.replace("_standardized", "_list_dict")}'
    #     processed_datasets[idx].push_to_hub(new_name)
    #
    # # alpaca datasets generation
    # processed_datasets = [format(dataset, alpaca_template) for dataset in list_of_datasets]
    # for idx, dataset_name in enumerate(list_of_datasets):
    #     new_name = f'{dataset_name.replace("_standardized", "_alpaca")}'
    #     processed_datasets[idx].push_to_hub(new_name)
    list_of_datasets = [
        f"HydraLM/GPTeacher_{subset}_standardized"
        for subset in ["codegen", "roleplay", "toolformer"]
    ]
    list_of_datasets += [
        "HydraLM/GPTeacher-General-Instruct_standardized",
        "HydraLM/airoboros-gpt4-1.4_standardized",
        "HydraLM/GPT4-LLM-Cleaned_standardized",
        "HydraLM/WizardLM_evol_instruct_V2_196k_standardized",
        "HydraLM/CodeAlpaca-20k_standardized",
        "HydraLM/unnatural-instructions_standardized",
    ]

    # combined cluster generation
    processed_datasets = [
        format(dataset, cluster_template) for dataset in list_of_datasets
    ]
    combined = combine_datasets(processed_datasets, list_of_datasets)
    combined.push_to_hub("HydraLM/combined_for_clustering")
