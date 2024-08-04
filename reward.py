import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from dataset_utils import get_pairs_for_reward


class RewardDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.prepare_data()

    def prepare_data(self):
        data = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": []
        }

        for pair in self.pairs:
            pos, neg = pair

            pos_enc = self.tokenizer(pos, truncation=True, padding='max_length', max_length=self.max_length)
            neg_enc = self.tokenizer(neg, truncation=True, padding='max_length', max_length=self.max_length)

            data["input_ids_chosen"].append(pos_enc["input_ids"])
            data["attention_mask_chosen"].append(pos_enc["attention_mask"])
            data["input_ids_rejected"].append(neg_enc["input_ids"])
            data["attention_mask_rejected"].append(neg_enc["attention_mask"])

        return data

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.data.items()}


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train_reward_model(pairs: List[Tuple[str, str]], config):
    # Загрузка параметров из конфигурации
    reward_config_params = config['reward']

    model_name = reward_config_params['model_name']
    output_dir = reward_config_params['output_dir']
    epochs = reward_config_params['epochs']
    batch_size = reward_config_params['batch_size']
    learning_rate = float(reward_config_params['learning_rate'])
    max_length = reward_config_params['max_length']
    logging_steps = reward_config_params['logging_steps']

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Подготовка данных
    train_dataset = RewardDataset(pairs, tokenizer, max_length)

    # Настройка параметров обучения
    reward_config = RewardConfig(
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        output_dir=output_dir,
        logging_steps=logging_steps,
        max_length=max_length,
        remove_unused_columns=False
    )

    if len(train_dataset) == 0:
        raise ValueError("Размер датасета равен 0. Проверьте входные данные.")

    # Обучение модели
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=reward_config
    )

    trainer.train()
    trainer.save_model(output_dir)
    return model


if __name__ == "__main__":
    # Загрузка конфигурации
    config_path = 'config.yaml'
    config = load_config(config_path)
    pairs = get_pairs_for_reward(config)
    model = train_reward_model(pairs, config)
