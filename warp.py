from tqdm import tqdm
import yaml
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification

from dataset_utils import get_prompts_warp


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def initialize_model_weights(model_name):
    """
    Функция для инициализации весов модели.

    Аргументы:
    model_name (str): Название предобученной модели.

    Возвращает:
    dict: Словарь с начальными весами модели.
    dict: Словарь с весами \(\theta_{\text{sft}}\).
    """
    # Загрузка модели
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Получение начальных весов
    theta_sft = model.state_dict().copy()  # \(\theta_{\text{sft}}\)
    theta_init = theta_sft.copy()  # \(\theta_{\text{init}}\)

    return theta_init, theta_sft


def slerp(theta_init, theta_m_list, lambda_val=0.5):
    """
    Сферическая линейная интерполяция (SLERP) для объединения весов.

    Аргументы:
    theta_init (dict): Начальные веса модели.
    theta_m_list (list): Список весов моделей для объединения.
    lambda_val (float): Коэффициент интерполяции.

    Возвращает:
    dict: Объединенные веса модели.
    """
    theta_slerp = {}
    for name in theta_init:
        # Векторы задач
        delta1 = theta_m_list[0][name] - theta_init[name]
        delta2 = theta_m_list[1][name] - theta_init[name]

        # Угол между векторами задач
        norm_delta1 = torch.norm(delta1)
        norm_delta2 = torch.norm(delta2)
        cos_omega = torch.dot(delta1.flatten(), delta2.flatten()) / (norm_delta1 * norm_delta2)
        omega = torch.acos(cos_omega)

        # SLERP интерполяция
        if omega != 0:
            sin_omega = torch.sin(omega)
            slerp_val = (torch.sin((1 - lambda_val) * omega) / sin_omega) * delta1 + (
                    torch.sin(lambda_val * omega) / sin_omega) * delta2
        else:
            slerp_val = delta1 * (1 - lambda_val) + delta2 * lambda_val

        theta_slerp[name] = theta_init[name] + slerp_val

    return theta_slerp


def liti(theta_init, theta_slerp, eta):
    """
    Линейная интерполяция к инициализации (LITI).

    Аргументы:
    theta_init (dict): Начальные веса модели.
    theta_slerp (dict): Веса модели после применения SLERP.
    eta (float): Интерполяционный коэффициент.

    Возвращает:
    dict: Интерполированные веса модели.
    """
    theta_liti = {}
    for name in theta_init:
        theta_liti[name] = (1 - eta) * theta_init[name] + eta * theta_slerp[name]
    return theta_liti


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
    # Декодирование списка индексов токенов обратно в текст
    y_text = reward_tokenizer.decode(y, skip_special_tokens=True)

    inputs = reward_tokenizer(x + y_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    outputs = reward_model(**inputs)
    reward = outputs.logits.item()
    return reward


def warp_algorithm(config):
    """
    Основная функция для выполнения алгоритма WARP.

    Аргументы:
    config (dict): Словарь с конфигурацией для WARP и модели наград.

    Возвращает:
    dict: Итоговые веса модели.
    """
    # Загрузка конфигурации
    warp_config = config['warp']
    reward_config = config['reward']
    X = get_prompts_warp(config)
    # Параметры для WARP
    model_name = config['model_name']
    I = warp_config['I']
    M = warp_config['M']
    T = warp_config['T']
    mu = warp_config['mu']
    eta = warp_config['eta']
    beta = warp_config['beta']
    batch_size = warp_config['batch_size']
    optimizer_type = warp_config['optimizer']['type']
    lr = float(warp_config['optimizer']['lr'])
    max_length = warp_config.get('max_length', 512)

    # Инициализация начальных весов
    theta_init, theta_sft = initialize_model_weights(model_name)

    # Инициализация токенизатора
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Выбор оптимизатора
    if optimizer_type == 'adam':
        Opt = torch.optim.Adam
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Загрузка модели и токенизатора наград
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_config['output_dir'], num_labels=1)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_config['output_dir'])

    # Основной цикл по итерациям
    for i in tqdm(range(1, I + 1)):
        # Список для хранения весов всех запусков RL
        theta_m_list = []

        # Цикл по запускам RL
        for m in range(1, M + 1):
            # Определение начальных весов для каждого запуска
            theta_m = theta_init.copy()
            theta_m_ema = theta_init.copy()

            # Определение модели и оптимизатора для текущего запуска
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.load_state_dict(theta_m)
            optimizer = Opt(model.parameters(), lr=lr)

            # Цикл по шагам обучения
            for t in range(1, T + 1):
                # Батчи данных
                for batch_start in range(0, len(X), batch_size):
                    batch = X[batch_start:batch_start + batch_size]

                    # Генерация завершений и вычисление наград для батча
                    for x in batch:
                        inputs = tokenizer(x, return_tensors='pt', truncation=True, max_length=max_length)
                        generated_outputs = model.generate(**inputs, max_length=max_length)
                        y = generated_outputs[0].tolist()

                        # Получение логитов для вычисления регуляризованной награды
                        outputs = model(**inputs)
                        logits = outputs.logits

                        # Вычисление регуляризованной награды
                        reward = reward_function(reward_model, reward_tokenizer, x, y, max_length)
                        log_prob = F.log_softmax(logits, dim=-1)
                        kl_divergence = F.kl_div(log_prob, torch.softmax(logits, dim=-1), reduction='batchmean')
                        r_beta = reward - beta * kl_divergence.mean()

                        # Обновление весов
                        optimizer.zero_grad()
                        r_beta.backward(retain_graph=True)
                        optimizer.step()

                        # Обновление весов EMA
                        for name in theta_m:
                            theta_m_ema[name] = (1 - mu) * theta_m_ema[name] + mu * theta_m[name]

            # Добавление текущих весов в список
            theta_m_list.append(theta_m)

        # Объединение весов после всех шагов обучения
        theta_slerp = slerp(theta_init, theta_m_list)
        # Обновление начальных весов с использованием LITI
        theta_init = liti(theta_init, theta_slerp, eta)
    return theta_init


if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)
    final_weights = warp_algorithm(config)
    torch.save(final_weights, "final_weights.pth")
