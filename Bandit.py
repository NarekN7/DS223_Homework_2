from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import os


class Bandit(ABC):
    """
    Abstract base class for multi-armed bandirlgorithms.
    
    All bandit implementations must inherit from this class.
    """
    @abstractmethod
    def __init__(self, p):
        """Initialize bandit with true reward probability."""
        pass

    @abstractmethod
    def __repr__(self):
        """String representation of bandit."""
        pass

    @abstractmethod
    def pull(self):
        """Pull the bandit and receive reward."""
        pass

    @abstractmethod
    def update(self):
        """Update bandit statistics based on reward."""
        pass

    @abstractmethod
    def experiment(self):
        """Run bandit experiment."""
        pass

    @abstractmethod
    def report(self):
        """Report bandit statistics."""
        pass


class Visualization():
    """
    Visualization class for bandit algorithm results.
    
    Attributes:
        bandit_results (List[dict]): List of bandit experiment results.
    """
    
    def __init__(self, bandit_results: List[dict]):
        """
        Initialize visualization with bandit results.
        
        Args:
            bandit_results (List[dict]): List of dictionaries containing results.
        """
        self.bandit_results = bandit_results
    
    def plot1(self):
        """
        Plot individual bandit performance over time.
        
        Generates linear and log scale plots showing estimated means.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        for bandit in self.bandit_results:
            estimated_means = bandit['estimated_means']
            trials = range(1, len(estimated_means) + 1)
            axes[0].plot(trials, estimated_means, label=f'Bandit {bandit["bandit_id"]} (p={bandit["true_mean"]})')
        
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Estimated Mean Reward')
        axes[0].set_title('Bandit Performance - Linear Scale')
        axes[0].legend()
        axes[0].grid(True)
        
        for bandit in self.bandit_results:
            estimated_means = bandit['estimated_means']
            trials = range(1, len(estimated_means) + 1)
            axes[1].plot(trials, estimated_means, label=f'Bandit {bandit["bandit_id"]} (p={bandit["true_mean"]})')
        
        axes[1].set_xlabel('Trial (Log Scale)')
        axes[1].set_ylabel('Estimated Mean Reward')
        axes[1].set_title('Bandit Performance - Log Scale')
        axes[1].set_xscale('log')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('bandit_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot2(self, eg_results: dict, ts_results: dict):
        """

        Args:
          eg_results: dict: 
          ts_results: dict: 

        Returns:

        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        

        axes[0].plot(eg_results['cumulative_rewards'], label='Epsilon Greedy', linewidth=2)
        axes[0].plot(ts_results['cumulative_rewards'], label='Thompson Sampling', linewidth=2)
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].set_title('Cumulative Rewards Comparison')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(eg_results['cumulative_regrets'], label='Epsilon Greedy', linewidth=2)
        axes[1].plot(ts_results['cumulative_regrets'], label='Thompson Sampling', linewidth=2)
        axes[1].set_xlabel('Trial')
        axes[1].set_ylabel('Cumulative Regret')
        axes[1].set_title('Cumulative Regrets Comparison')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy Algorithm Implementation
    
    Balances exploration and exploitation using an epsilon probability:
    - With probability epsilon: explores randomly
    - With probability (1-epsilon): exploits the best estimated arm
    - Epsilon decays as 1/t where t is trial number
    """
    
    def __init__(self, p):
        """
        Initialize bandit with true reward probability.
        
        Args:
            p: True mean reward of this bandit arm.
        """
        self.p = p
        self.p_estimate = 0.0
        self.N = 0
        self.epsilon = 1.0
        self.rewards = []
        self.last_reward = 0.0
        
    def __repr__(self):
        return f'EpsilonGreedy(p={self.p}, p_estimate={self.p_estimate:.2f}, N={self.N})'
    
    def pull(self):
        """
        Pull the bandit arm and receive a reward.
        
        Rewards are drawn from a Gaussian distribution centered at the true mean.
        Returns: Sampled reward value.
        """
        reward = np.random.normal(self.p, scale=1.0)
        self.last_reward = reward
        return reward
    
    def update(self):
        """
        Update bandit statistics after receiving a reward.
        
        Updates:
        - Running average of rewards (p_estimate)
        - Epsilon value using 1/t decay for decreasing exploration over time
        """
        reward = self.last_reward
        self.N += 1
        self.rewards.append(reward)
        self.p_estimate = (self.p_estimate * (self.N - 1) + reward) / self.N
        self.epsilon = 1.0 / self.N if self.N > 0 else 1.0
    
    def experiment(self):
        
        pass
    
    def report(self):
        

        if self.N == 0:
            print(f"EpsilonGreedy Bandit (p={self.p}): No pulls yet")
            return
        
        avg_reward = self.p_estimate
        optimal_mean = 4
        avg_regret = optimal_mean - self.p
        
        print(f"EpsilonGreedy Bandit (p={self.p}): N={self.N}, "
              f"Avg Reward={avg_reward:.4f}, Epsilon={self.epsilon:.4f}")


class ThompsonSampling(Bandit):
    """
    Thompson Sampling Algorithm Implementation
    
    Bayesian approach for multi-armed bandits:
    - Maintains posterior distribution for each arm's reward
    - Samples from posteriors to select arms (optimism in uncertainty)
    - Updates posteriors using Bayesian inference after each pull
    - Naturally balances exploration and exploitation
    """
    
    def __init__(self, p, tau=1.0, mu_0=0.0, lambda_0=1.0):
        """
        Initialize bandit with Bayesian prior parameters.
        
        Args:
            p: True mean reward (unknown in practice)
            tau: Known precision (inverse variance) of rewards
            mu_0: Prior mean for Bayesian inference
            lambda_0: Prior precision for Bayesian inference
        """
        self.p = p
        self.tau = tau
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        
        self.mu = mu_0 
        self.lambda_posterior = lambda_0
        self.N = 0
        self.rewards = []
        self.last_reward = 0.0
        
    def __repr__(self):
        return f'ThompsonSampling(p={self.p}, mu={self.mu:.2f}, lambda={self.lambda_posterior:.2f}, N={self.N})'
    
    def pull(self):
        """
        Pull the bandit arm and receive a reward.
        
        Returns: Reward sampled from Gaussian distribution with known precision.
        """
        reward = np.random.normal(self.p, scale=1.0/np.sqrt(self.tau))
        self.last_reward = reward
        return reward
    
    def update(self):
        """
        Update posterior distribution using Bayesian inference.
        
        For Gaussian likelihood with known precision tau and Gaussian prior:
        - Posterior precision increases linearly with number of observations
        - Posterior mean is precision-weighted combination of prior and data
        - Updates are conjugate, maintaining Gaussian posteriors
        """
        reward = self.last_reward
        self.N += 1
        self.rewards.append(reward)
        
        self.lambda_posterior = self.lambda_0 + self.N * self.tau
        sum_rewards = sum(self.rewards)
        self.mu = (self.lambda_0 * self.mu_0 + self.tau * sum_rewards) / self.lambda_posterior
    
    def sample(self):
        """
        Sample from the posterior distribution of mean reward.
        
        Used by Thompson Sampling to select arms:
        - Returns a sample representing our belief about the arm's true reward
        - Arms with higher uncertainty get explored more due to wider posteriors
        - Arms with high posterior means get exploited more
        """
        posterior_var = 1.0 / self.lambda_posterior
        return np.random.normal(self.mu, scale=np.sqrt(posterior_var))
    
    def experiment(self):
        """Placeholder for experiment method.
        This is called at the bandit level, but the actual experiment logic
        is in the multi-arm bandit context.

        Args:

        Returns:

        """
        pass
    
    def report(self):
        
        if self.N == 0:
            print(f"ThompsonSampling Bandit (p={self.p}): No pulls yet")
            return
        
        avg_reward = self.mu
        optimal_mean = 4
        
        print(f"ThompsonSampling Bandit (p={self.p}): N={self.N}, "
              f"Posterior Mean={self.mu:.4f}, Lambda={self.lambda_posterior:.4f}")




def run_epsilon_greedy_experiment(bandit_rewards: List[float], num_trials: int) -> Tuple[dict, pd.DataFrame]:
    """
    Run Epsilon-Greedy experiment on multiple bandit arms.
    
    Implementation details:
    - Uses epsilon-greedy strategy with 1/t decay
    - Tracks cumulative rewards and regrets
    - Returns results with estimated means for visualization
    
    Args:
        bandit_rewards: True mean rewards for each bandit arm
        num_trials: Number of trials to run
        
    Returns:
        Tuple of (results dictionary, pandas DataFrame with trial data)
    """
    logger.info(f"Starting Epsilon Greedy experiment with {num_trials} trials")
    
    bandits = [EpsilonGreedy(p) for p in bandit_rewards]
    
    rewards_data = []
    cumulative_reward = 0
    cumulative_regret = 0
    optimal_reward = max(bandit_rewards)
    
    cumulative_rewards_history = []
    cumulative_regrets_history = []
    estimated_means_history = [[] for _ in bandit_rewards]
    
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
            'Algorithm': 'EpsilonGreedy'
        })
        
        cumulative_reward += reward
        cumulative_regret += optimal_reward - bandits[i].p
        
        cumulative_rewards_history.append(cumulative_reward)
        cumulative_regrets_history.append(cumulative_regret)
        
        for bandit in bandits:
            bandit.epsilon = 1.0 / (t + 1)
        
        for j, bandit in enumerate(bandits):
            estimated_means_history[j].append(bandit.p_estimate)
    
    results = {
        'cumulative_rewards': cumulative_rewards_history,
        'cumulative_regrets': cumulative_regrets_history,
        'estimated_means': estimated_means_history,
        'bandit_results': [
            {'bandit_id': i, 'true_mean': bandit_rewards[i], 'estimated_means': estimated_means_history[i]}
            for i in range(len(bandits))
        ]
    }
    
    df = pd.DataFrame(rewards_data)
    logger.info(f"Epsilon Greedy experiment completed. Final cumulative reward: {cumulative_reward:.2f}")
    
    return results, df


def run_thompson_sampling_experiment(bandit_rewards: List[float], num_trials: int, tau: float = 1.0) -> Tuple[dict, pd.DataFrame]:
    """
    Run Thompson Sampling experiment on multiple bandit arms.
    
    Implementation details:
    - Uses Bayesian posterior sampling to select arms
    - Maintains Gaussian posteriors for each arm's reward distribution
    - Naturally balances exploration of uncertain arms vs exploitation of known good arms
    
    Args:
        bandit_rewards: True mean rewards for each bandit arm
        num_trials: Number of trials to run
        tau: Known precision (inverse variance) of reward distributions
        
    Returns:
        Tuple of (results dictionary, pandas DataFrame with trial data)
    """
    logger.info(f"Starting Thompson Sampling experiment with {num_trials} trials")
    
    bandits = [ThompsonSampling(p, tau=tau) for p in bandit_rewards]
    
    rewards_data = []
    cumulative_reward = 0
    cumulative_regret = 0
    optimal_reward = max(bandit_rewards)
    
    cumulative_rewards_history = []
    cumulative_regrets_history = []
    estimated_means_history = [[] for _ in bandit_rewards]
    
    for t in range(num_trials):
        samples = [bandit.sample() for bandit in bandits]
        i = np.argmax(samples)
        
        reward = bandits[i].pull()
        bandits[i].update()
        
        rewards_data.append({
            'Bandit': i,
            'Reward': reward,
            'Algorithm': 'ThompsonSampling'
        })
        
        cumulative_reward += reward
        cumulative_regret += optimal_reward - bandits[i].p
        
        cumulative_rewards_history.append(cumulative_reward)
        cumulative_regrets_history.append(cumulative_regret)
        
        for j, bandit in enumerate(bandits):
            estimated_means_history[j].append(bandit.mu)
    
    results = {
        'cumulative_rewards': cumulative_rewards_history,
        'cumulative_regrets': cumulative_regrets_history,
        'estimated_means': estimated_means_history,
        'bandit_results': [
            {'bandit_id': i, 'true_mean': bandit_rewards[i], 'estimated_means': estimated_means_history[i]}
            for i in range(len(bandits))
        ]
    }
    
    df = pd.DataFrame(rewards_data)
    logger.info(f"Thompson Sampling experiment completed. Final cumulative reward: {cumulative_reward:.2f}")
    
    return results, df


def comparison():
    """
    Main comparison function for Epsilon-Greedy vs Thompson Sampling.
    
    Runs both algorithms, generates visualizations, saves results to CSV,
    and prints cumulative rewards and regrets for comparison.
    """
    bandit_rewards = [1, 2, 3, 4]
    num_trials = 20000
    
    logger.info("="*60)
    logger.info("A/B Testing Experiment: Epsilon Greedy vs Thompson Sampling")
    logger.info("="*60)
    
    eg_results, eg_df = run_epsilon_greedy_experiment(bandit_rewards, num_trials)
    
    ts_results, ts_df = run_thompson_sampling_experiment(bandit_rewards, num_trials, tau=1.0)
    
    combined_df = pd.concat([eg_df, ts_df], ignore_index=True)
    
    csv_filename = 'bandit_results.csv'
    combined_df.to_csv(csv_filename, index=False)
    logger.info(f"Results saved to {csv_filename}")
    
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    print(f"\nEpsilon Greedy:")
    print(f"  Total Cumulative Reward: {eg_results['cumulative_rewards'][-1]:.2f}")
    print(f"  Final Cumulative Regret: {eg_results['cumulative_regrets'][-1]:.2f}")
    print(f"  Average Reward: {eg_results['cumulative_rewards'][-1]/num_trials:.4f}")
    
    print(f"\nThompson Sampling:")
    print(f"  Total Cumulative Reward: {ts_results['cumulative_rewards'][-1]:.2f}")
    print(f"  Final Cumulative Regret: {ts_results['cumulative_regrets'][-1]:.2f}")
    print(f"  Average Reward: {ts_results['cumulative_rewards'][-1]/num_trials:.4f}")
    
    logger.info("Creating visualizations...")
    
    viz = Visualization(eg_results['bandit_results'])
    viz.plot1()
    
    viz.plot2(eg_results, ts_results)
    
    print("\nVisualizations saved:")
    print("  - bandit_performance.png")
    print("  - algorithm_comparison.png")
    print("\n" + "="*60)

if __name__=='__main__':
    comparison()