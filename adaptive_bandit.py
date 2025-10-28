import numpy as np
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from Bandit import Bandit


class AdaptiveEpsilonGreedy(Bandit):
    """
    Adaptive Epsilon-Greedy with Exponential Decay.
    
    Enhancement over standard epsilon-greedy:
    - Uses exponential decay: epsilon(t) = epsilon_0 * e^(-kt)
    - Provides more controlled exploration schedule
    - Better for uncertain or non-stationary environments
    - k parameter controls decay rate (smaller = slower decay, more exploration)
    """
    
    def __init__(self, p, epsilon_0=1.0, k=0.005):
        """
        Initialize adaptive epsilon-greedy bandit.
        
        Args:
            p: True mean reward of bandit arm
            epsilon_0: Initial exploration probability
            k: Decay rate parameter (smaller = slower decay)
        """
        self.p = p
        self.p_estimate = 0.0
        self.N = 0
        self.epsilon_0 = epsilon_0
        self.k = k
        self.epsilon = epsilon_0
        self.rewards = []
        self.last_reward = 0.0
        
    def __repr__(self):
        return f'AdaptiveEpsilonGreedy(p={self.p}, p_estimate={self.p_estimate:.2f}, N={self.N})'
    
    def pull(self):
        """Pull bandit arm and receive reward."""
        reward = np.random.normal(self.p, scale=1.0)
        self.last_reward = reward
        return reward
    
    def update(self):
        """
        Update bandit statistics with exponential epsilon decay.
        
        Key difference from standard epsilon-greedy:
        - Uses exponential function for epsilon decay instead of 1/t
        - Allows more fine-tuned control over exploration schedule
        - Maintains higher epsilon longer for uncertain environments
        """
        reward = self.last_reward
        self.N += 1
        self.rewards.append(reward)
        self.p_estimate = (self.p_estimate * (self.N - 1) + reward) / self.N
        self.epsilon = self.epsilon_0 * np.exp(-self.k * self.N)
    
    def experiment(self):
        
        pass
    
    def report(self):
        
        if self.N == 0:
            print(f"AdaptiveEpsilonGreedy Bandit (p={self.p}): No pulls yet")
            return
        print(f"AdaptiveEpsilonGreedy Bandit (p={self.p}): N={self.N}, "
              f"Avg Reward={self.p_estimate:.4f}, Epsilon={self.epsilon:.6f}")


def run_custom_sampling_experiment(bandit_rewards: List[float], num_trials: int, epsilon_0: float = 1.0, k: float = 0.005) -> Tuple[dict, pd.DataFrame]:
    """
    Run Adaptive Epsilon-Greedy experiment with exponential decay.
    
    Compares adaptive epsilon-greedy using exponential decay formula:
    epsilon(t) = epsilon_0 * e^(-kt)
    
    Args:
        bandit_rewards: True mean rewards for each arm
        num_trials: Number of trials
        epsilon_0: Initial exploration probability
        k: Decay rate (smaller = more exploration)
        
    Returns:
        Results dictionary and DataFrame with trial data
    """
    logger.info(f"Starting Adaptive Epsilon Greedy experiment with {num_trials} trials")
    logger.info(f"Parameters: epsilon_0={epsilon_0}, k={k}")
    
    bandits = [AdaptiveEpsilonGreedy(p, epsilon_0=epsilon_0, k=k) for p in bandit_rewards]
    
    rewards_data = []
    cumulative_reward = 0
    cumulative_regret = 0
    optimal_reward = max(bandit_rewards)
    
    cumulative_rewards_history = []
    cumulative_regrets_history = []
    estimated_means_history = [[] for _ in bandit_rewards]
    epsilon_history = []
    
    for t in range(num_trials):
        if np.random.random() < bandits[0].epsilon:
            i = np.random.randint(len(bandits))
        else:
            i = np.argmax([b.p_estimate for b in bandits])
        
        reward = bandits[i].pull()
        bandits[i].update()
        
        rewards_data.append({
            'Bandit': i,
            'Reward': reward,
            'Algorithm': 'AdaptiveEpsilonGreedy'
        })
        
        cumulative_reward += reward
        cumulative_regret += optimal_reward - bandits[i].p
        
        cumulative_rewards_history.append(cumulative_reward)
        cumulative_regrets_history.append(cumulative_regret)
        epsilon_history.append(bandits[0].epsilon)
        
        for j, bandit in enumerate(bandits):
            estimated_means_history[j].append(bandit.p_estimate)
    
    results = {
        'cumulative_rewards': cumulative_rewards_history,
        'cumulative_regrets': cumulative_regrets_history,
        'epsilon_history': epsilon_history,
        'estimated_means': estimated_means_history,
        'bandit_results': [
            {'bandit_id': i, 'true_mean': bandit_rewards[i], 'estimated_means': estimated_means_history[i]}
            for i in range(len(bandits))
        ]
    }
    
    df = pd.DataFrame(rewards_data)
    logger.info(f"Adaptive Epsilon Greedy experiment completed. Final cumulative reward: {cumulative_reward:.2f}")
    
    return results, df


def run_parallel_experiments(num_runs: int = 100, num_trials: int = 20000, bandit_rewards: List[float] = [1, 2, 3, 4]):
    """
    Run multiple parallel experiments for statistical robustness.
    
    Purpose:
    - Provides statistical confidence in results
    - Shows variance across different runs
    - Generates mean and standard deviation for visualizations
    
    Args:
        num_runs: Number of independent experimental runs
        num_trials: Trials per run
        bandit_rewards: Reward values for bandit arms
        
    Returns:
        Dictionary with mean and std statistics
    """
    logger.info(f"Running {num_runs} parallel experiments with {num_trials} trials each...")
    
    all_cumulative_rewards = []
    all_cumulative_regrets = []
    
    for run in range(num_runs):
        if run % 10 == 0:
            logger.info(f"Completed {run}/{num_runs} runs...")
        
        results, _ = run_custom_sampling_experiment(bandit_rewards, num_trials, epsilon_0=1.0, k=0.005)
        all_cumulative_rewards.append(results['cumulative_rewards'])
        all_cumulative_regrets.append(results['cumulative_regrets'])
    
    all_cumulative_rewards = np.array(all_cumulative_rewards)
    all_cumulative_regrets = np.array(all_cumulative_regrets)
    
    rewards_mean = np.mean(all_cumulative_rewards, axis=0)
    rewards_std = np.std(all_cumulative_rewards, axis=0)
    regrets_mean = np.mean(all_cumulative_regrets, axis=0)
    regrets_std = np.std(all_cumulative_regrets, axis=0)
    
    logger.info("Parallel experiments completed!")
    
    return {
        'rewards_mean': rewards_mean,
        'rewards_std': rewards_std,
        'regrets_mean': regrets_mean,
        'regrets_std': regrets_std,
        'num_runs': num_runs
    }


def plot_parallel_results(parallel_results: dict):
    """
    Visualize parallel experiment results with confidence bands.
    
    Args:
        parallel_results: Dictionary with mean and std statistics
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    num_trials = len(parallel_results['rewards_mean'])
    trials = range(num_trials)
    
    rewards_mean = parallel_results['rewards_mean']
    rewards_std = parallel_results['rewards_std']
    axes[0].plot(trials, rewards_mean, label='Mean Reward', linewidth=2, color='blue')
    axes[0].fill_between(trials, rewards_mean - rewards_std, rewards_mean + rewards_std, 
                         alpha=0.3, label='±1 Std Dev', color='blue')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title(f'Cumulative Rewards (Mean ± Std) - {parallel_results["num_runs"]} Runs')
    axes[0].legend()
    axes[0].grid(True)
    
    regrets_mean = parallel_results['regrets_mean']
    regrets_std = parallel_results['regrets_std']
    axes[1].plot(trials, regrets_mean, label='Mean Regret', linewidth=2, color='red')
    axes[1].fill_between(trials, regrets_mean - regrets_std, regrets_mean + regrets_std,
                         alpha=0.3, label='±1 Std Dev', color='red')
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Cumulative Regret')
    axes[1].set_title(f'Cumulative Regrets (Mean ± Std) - {parallel_results["num_runs"]} Runs')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('adaptive_parallel_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Parallel results visualization saved to adaptive_parallel_results.png")


if __name__ == '__main__':
    logger.info("Testing Adaptive Epsilon Greedy implementation...")
    
    bandit_rewards = [1, 2, 3, 4]
    num_trials = 20000
    
    results, df = run_custom_sampling_experiment(bandit_rewards, num_trials)
    
    df.to_csv('adaptive_bandit_results.csv', index=False)
    print(f"\nAdaptive Epsilon Greedy Results:")
    print(f"  Total Cumulative Reward: {results['cumulative_rewards'][-1]:.2f}")
    print(f"  Final Cumulative Regret: {results['cumulative_regrets'][-1]:.2f}")
    print(f"  Average Reward: {results['cumulative_rewards'][-1]/num_trials:.4f}")
    print(f"\nResults saved to adaptive_bandit_results.csv")
    
    parallel_results = run_parallel_experiments(num_runs=10, num_trials=num_trials, bandit_rewards=bandit_rewards)
    plot_parallel_results(parallel_results)

