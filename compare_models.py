import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification

from prod.dataset_utils import get_prompts_warp


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def reward_function(reward_model, reward_tokenizer, x, y, max_length=512):
    """
    Функция награды, использующая предобученную модель наград.

    Аргументы:
    reward_model (torch.nn.Module): Модель наград.
    reward_tokenizer (transformers.Tokenizer): Токенизатор для модели наград.
    x (str): Исходный текст.
    y (list): Список индексов токенов сгенерированного текста.
    max_length (int): Максимальная длина входной последовательности.

    Возвращает:
    float: Награда.
    """
    y_text = reward_tokenizer.decode(y, skip_special_tokens=True)
    inputs = reward_tokenizer(x + y_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    outputs = reward_model(**inputs)
    reward = outputs.logits.item()
    return reward


def measure_performance(model, tokenizer, reward_model, reward_tokenizer, prompts, max_length=512):
    """
    Функция для измерения средней награды и KL-дивергенции.

    Аргументы:
    model (torch.nn.Module): Модель для генерации текстов.
    tokenizer (transformers.Tokenizer): Токенизатор для модели генерации.
    reward_model (torch.nn.Module): Модель наград.
    reward_tokenizer (transformers.Tokenizer): Токенизатор для модели наград.
    prompts (list): Список текстовых промптов.
    max_length (int): Максимальная длина входной последовательности.

    Возвращает:
    float: Средняя награда.
    float: Средняя KL-дивергенция.
    """
    total_reward = 0
    total_kl_divergence = 0
    count = 0

    for x in tqdm(prompts):
        inputs = tokenizer(x, return_tensors='pt', truncation=True, max_length=max_length)
        generated_outputs = model.generate(**inputs, max_length=max_length)
        y = generated_outputs[0].tolist()
        outputs = model(**inputs)
        logits = outputs.logits
        reward = reward_function(reward_model, reward_tokenizer, x, y, max_length)
        log_prob = F.log_softmax(logits, dim=-1)
        kl_divergence = F.kl_div(log_prob, torch.softmax(logits, dim=-1), reduction='batchmean')
        total_reward += reward
        total_kl_divergence += kl_divergence.item()
        count += 1

    avg_reward = total_reward / count
    avg_kl_divergence = total_kl_divergence / count

    return avg_reward, avg_kl_divergence


def compare_models(config):
    """
    Основная функция для сравнения выравненной модели и исходной SFT модели.

    Аргументы:
    config (dict): Словарь с конфигурацией для WARP и модели наград.
    """
    warp_config = config['warp']
    reward_config = config['reward']
    model_name = config['model_name']
    max_length = warp_config.get('max_length', 512)

    # Загрузка промптов
    prompts = get_prompts_warp(config)

    # Инициализация токенизатора
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Загрузка модели и токенизатора наград
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_config['output_dir'], num_labels=1)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_config['output_dir'])

    # Загрузка исходной SFT модели
    sft_model = GPT2LMHeadModel.from_pretrained(model_name)
    sft_model.eval()

    # Измерение производительности для SFT модели
    avg_reward_sft, avg_kl_sft = measure_performance(sft_model, tokenizer, reward_model, reward_tokenizer, prompts, max_length)
    print(f"Average Reward for SFT Model: {avg_reward_sft}, Average KL for SFT Model: {avg_kl_sft}")

    # Загрузка выравненной модели
    final_weights_path = "final_weights.pth"
    final_weights = torch.load(final_weights_path)
    aligned_model = GPT2LMHeadModel.from_pretrained(model_name)
    aligned_model.load_state_dict(final_weights)
    aligned_model.eval()

    # Измерение производительности для выравненной модели
    avg_reward_aligned, avg_kl_aligned = measure_performance(aligned_model, tokenizer, reward_model, reward_tokenizer, prompts, max_length)
    print(f"Average Reward for Aligned Model: {avg_reward_aligned}, Average KL for Aligned Model: {avg_kl_aligned}")

    # Вывод сравнения
    print(f"Comparison:\n"
          f"SFT Model - Average Reward: {avg_reward_sft}, Average KL: {avg_kl_sft}\n"
          f"Aligned Model - Average Reward: {avg_reward_aligned}, Average KL: {avg_kl_aligned}")


if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)
    compare_models(config)
