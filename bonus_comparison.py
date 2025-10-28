from Bandit import (
    run_epsilon_greedy_experiment,
    run_thompson_sampling_experiment,
    Visualization
)
from adaptive_bandit import (
    run_custom_sampling_experiment,
    run_parallel_experiments,
    plot_parallel_results
)
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def bonus_comparison():
    """
    Comprehensive comparison of all three bandit algorithms.
    
    Runs Epsilon Greedy, Thompson Sampling, and Adaptive Epsilon Greedy:
    - Generates separate CSV reports for each algorithm
    - Creates comparison visualizations
    - Runs parallel experiments for statistical validation
    - Prints comparative statistics
    """
    bandit_rewards = [1, 2, 3, 4]
    num_trials = 20000
    
    logger.info("="*70)
    logger.info("BONUS: Comprehensive Algorithm Comparison")
    logger.info("="*70)
    
    logger.info("\n" + "="*70)
    logger.info("1. Running Epsilon Greedy (1/t decay)")
    logger.info("="*70)
    eg_results, eg_df = run_epsilon_greedy_experiment(bandit_rewards, num_trials)
    
    logger.info("\n" + "="*70)
    logger.info("2. Running Thompson Sampling (Bayesian)")
    logger.info("="*70)
    ts_results, ts_df = run_thompson_sampling_experiment(bandit_rewards, num_trials, tau=1.0)
    
    logger.info("\n" + "="*70)
    logger.info("3. Running Adaptive Epsilon Greedy (Exponential Decay)")
    logger.info("="*70)
    adaptive_results, adaptive_df = run_custom_sampling_experiment(bandit_rewards, num_trials, epsilon_0=1.0, k=0.005)
    
    eg_df.to_csv('epsilon_greedy_report.csv', index=False)
    ts_df.to_csv('thompson_sampling_report.csv', index=False)
    adaptive_df.to_csv('adaptive_epsilon_report.csv', index=False)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*70)
    
    print(f"\n1. Epsilon Greedy (1/t decay):")
    print(f"   Total Cumulative Reward: {eg_results['cumulative_rewards'][-1]:.2f}")
    print(f"   Final Cumulative Regret: {eg_results['cumulative_regrets'][-1]:.2f}")
    print(f"   Average Reward: {eg_results['cumulative_rewards'][-1]/num_trials:.4f}")
    print(f"   Final Epsilon: 1/{num_trials} ≈ {1/num_trials:.6f}")
    
    print(f"\n2. Thompson Sampling (Bayesian):")
    print(f"   Total Cumulative Reward: {ts_results['cumulative_rewards'][-1]:.2f}")
    print(f"   Final Cumulative Regret: {ts_results['cumulative_regrets'][-1]:.2f}")
    print(f"   Average Reward: {ts_results['cumulative_rewards'][-1]/num_trials:.4f}")
    print(f"   Known Precision: τ = 1.0")
    
    print(f"\n3. Adaptive Epsilon Greedy (Exponential decay):")
    print(f"   Total Cumulative Reward: {adaptive_results['cumulative_rewards'][-1]:.2f}")
    print(f"   Final Cumulative Regret: {adaptive_results['cumulative_regrets'][-1]:.2f}")
    print(f"   Average Reward: {adaptive_results['cumulative_rewards'][-1]/num_trials:.4f}")
    print(f"   Final Epsilon: {adaptive_results['epsilon_history'][-1]:.6f}")
    print(f"   Decay: ε(t) = 1.0 · e^(-0.005t)")
    
    logger.info("\n" + "="*70)
    logger.info("Creating comparison visualizations...")
    logger.info("="*70)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    trials = range(num_trials)
    axes[0].plot(trials, eg_results['cumulative_rewards'], 
                 label='Epsilon Greedy (1/t)', linewidth=2, alpha=0.8)
    axes[0].plot(trials, ts_results['cumulative_rewards'], 
                 label='Thompson Sampling', linewidth=2, alpha=0.8)
    axes[0].plot(trials, adaptive_results['cumulative_rewards'], 
                 label='Adaptive Epsilon (e^(-kt))', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title('Cumulative Rewards Comparison - All Algorithms')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(trials, eg_results['cumulative_regrets'], 
                 label='Epsilon Greedy (1/t)', linewidth=2, alpha=0.8)
    axes[1].plot(trials, ts_results['cumulative_regrets'], 
                 label='Thompson Sampling', linewidth=2, alpha=0.8)
    axes[1].plot(trials, adaptive_results['cumulative_regrets'], 
                 label='Adaptive Epsilon (e^(-kt))', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('Cumulative Regret')
    axes[1].set_title('Cumulative Regrets Comparison - All Algorithms')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('bonus_comparison_all_algorithms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    trials_subset = range(0, min(5000, num_trials), 10)
    epsilon_1t = [1.0 / (t + 1) for t in trials_subset]
    epsilon_exp = [1.0 * np.exp(-0.005 * t) for t in trials_subset]
    
    plt.plot(trials_subset, epsilon_1t, label='1/t decay', linewidth=2)
    plt.plot(trials_subset, epsilon_exp, label='e^(-0.005t) decay', linewidth=2)
    plt.xlabel('Trial')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Decay Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('bonus_epsilon_decay_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n" + "="*70)
    print("VISUALIZATIONS SAVED:")
    print("  - bonus_comparison_all_algorithms.png")
    print("  - bonus_epsilon_decay_comparison.png")
    print("  - adaptive_parallel_results.png (will be generated next)")
    print("="*70)
    
    logger.info("\n" + "="*70)
    logger.info("Running parallel experiments for statistical robustness...")
    logger.info("="*70)
    logger.info("This may take a few minutes...")
    
    parallel_results = run_parallel_experiments(num_runs=50, num_trials=num_trials, 
                                               bandit_rewards=bandit_rewards)
    plot_parallel_results(parallel_results)
    
    print("\n" + "="*70)
    print("BONUS IMPLEMENTATION COMPLETE!")
    print("="*70)
    print("\nFiles generated:")
    print("  1. epsilon_greedy_report.csv")
    print("  2. thompson_sampling_report.csv")
    print("  3. adaptive_epsilon_report.csv")
    print("  4. bonus_comparison_all_algorithms.png")
    print("  5. bonus_epsilon_decay_comparison.png")
    print("  6. adaptive_parallel_results.png")
    print("\n" + "="*70)


if __name__ == '__main__':
    bonus_comparison()

