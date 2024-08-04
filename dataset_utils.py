import yaml
from typing import List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_pairs_for_reward(config) -> List[Tuple[str, str]]:
    """
    Получение пар (положительный комментарий, отрицательный комментарий) для модели наград.

    Аргументы:
    config (dict): Конфигурация, содержащая параметры для загрузки датасета.

    Возвращает:
    List[Tuple[str, str]]: Список пар (положительный комментарий, отрицательный комментарий).
    """
    dataset_name = config['data']['dataset_name']
    dataset = load_dataset(dataset_name)
    train_data = dataset['train']

    positive_comments = []
    negative_comments = []

    for example in train_data:
        if example['label'] == 1:
            positive_comments.append(example['text'])
        else:
            negative_comments.append(example['text'])

    # Составление пар
    # pairs = list(itertools.product(positive_comments, negative_comments))
    max_len = max(len(positive_comments), len(negative_comments))
    pairs = list(zip(positive_comments[:max_len], negative_comments[:max_len]))
    return pairs


def get_prompts_warp(config) -> List[str]:
    """
    Получение промптов для WARP.

    Аргументы:
    config (dict): Конфигурация, содержащая параметры для генерации промптов.

    Возвращает:
    List[str]: Список промптов.
    """
    dataset_name = config['data']['dataset_name']
    prompt_length_min = config['warp']['prompt_length_min']
    prompt_length_max = config['warp']['prompt_length_max']
    model_name = config['model_name']

    # Загрузка датасета и токенизатора
    dataset = load_dataset(dataset_name)
    train_data = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []

    for example in train_data:
        text = example['text']
        # Токенизация текста
        tokens = tokenizer.tokenize(text)
        if len(tokens) >= prompt_length_min:
            prompt = tokenizer.convert_tokens_to_string(tokens[:prompt_length_max])
            prompts.append(prompt)
        if len(prompts) >= 1000:
          break

    return prompts


def get_test_prompts(config):
    dataset_name = config['data']['dataset_name']
    prompt_length_min = config['warp']['prompt_length_min']
    prompt_length_max = config['warp']['prompt_length_max']

    dataset = load_dataset(dataset_name)
    test_data = dataset['test']

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    prompts = []
    for example in test_data:
        text = example['text']
        tokens = tokenizer.tokenize(text)
        if len(tokens) >= prompt_length_min:
            prompt = tokenizer.convert_tokens_to_string(tokens[:prompt_length_max])
            prompts.append(prompt)
        if len(prompts) >= 100:
            break

    return prompts


if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)
    pairs = get_pairs_for_reward(config)
    prompts = get_prompts_warp(config)
