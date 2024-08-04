import yaml
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset_utils import get_test_prompts
from warp import load_config, initialize_model_weights, reward_function, warp_algorithm


def measure_performance(config, model_weights, prompts):
    reward_config = config['reward']
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_config['output_dir'], num_labels=1)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_config['output_dir'])

    model_name = config['model_name']
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.load_state_dict(model_weights)

    total_reward = 0
    total_kl_divergence = 0
    count = 0
    max_length = config['warp'].get('max_length', 512)

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


def plot_results(eta_values, rewards, kl_divergences):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Eta')
    ax1.set_ylabel('Average Reward', color=color)
    ax1.plot(eta_values, rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average KL Divergence', color=color)
    ax2.plot(eta_values, kl_divergences, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Average Reward and KL Divergence vs Eta')
    plt.show()


if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)

    # Получение 100 промптов из тестовой подвыборки
    prompts = get_test_prompts(config)

    # Инициализация начальных весов
    model_name = config['model_name']
    theta_init, theta_sft = initialize_model_weights(model_name)
    avg_reward_sft, avg_kl_sft = measure_performance(config, theta_sft, prompts)
    print(f"Average Reward for θ_sft: {avg_reward_sft}, Average KL for θ_sft: {avg_kl_sft}")

    # Измерение производительности для обученной модели с разными значениями eta
    eta_values = [0.1, 0.5, 0.9]
    rewards = []
    kl_divergences = []

    for eta in eta_values:
        config['warp']['eta'] = eta
        final_weights = warp_algorithm(config)
        avg_reward, avg_kl_divergence = measure_performance(config, final_weights, prompts)
        rewards.append(avg_reward)
        kl_divergences.append(avg_kl_divergence)
        print(f"Average Reward for eta={eta}: {avg_reward}, Average KL for eta={eta}: {avg_kl_divergence}")

    # Построение графика
    plot_results(eta_values, rewards, kl_divergences)
