import json
import numpy as np
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from game.prioritized_game import PrioritizedDatingGame
from prioritized_env import env
from reward.coefficient_reward import CoefficientReward
from visualizations.analytics import save_analytics
from visualizations.visualizations import create_all_visualizations
from reinforcement_learning_algos.PPO import create_individual_ppo_agents as create_ppo_agents
from reinforcement_learning_algos.MC_PPO import create_individual_ppo_agents as create_mc_ppo_agents
from reinforcement_learning_algos.PPO import IndividualPPOAgent

# Configuration flag for PPO type
USE_MONTE_CARLO = True  # Set to True to use MC_PPO, False to use regular PPO

def load_configs(config_path: str) -> list:
    """Load simulation configurations from JSON file"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data['configs']

def setup_logging(log_dir="runs/randomized"):
    """Set up the logging environment for TensorBoard"""
    return SummaryWriter(log_dir)

def collect_episode_metrics(game: PrioritizedDatingGame, agents: Dict[str, IndividualPPOAgent], total_reward: float) -> Dict:
    """Collects analytics for the episode based on game state and agent performance."""
    matches = game.get_all_matches()
    traits = game.get_all_traits()
    total_matches = sum(len(matches[agent]) for agent in agents) // 2
    total_swipes = sum(len(swipe_set) for swipe_set in game.swipes.values())
    avg_reward = total_reward / len(agents)
    
    disparities = [
        abs(np.mean(traits[player]) - np.mean(traits[match]))
        for player, player_matches in matches.items()
        for match in player_matches
    ]
    avg_match_disparity = np.mean(disparities) if disparities else 0
    
    return {
        'Simulation': {
            'total_matches': total_matches,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'avg_match_disparity': avg_match_disparity,
            'match_swipe_ratio': total_matches / (total_swipes / 2) if total_matches > 0 else 0
        },
        'Men': {
            'men_matches': {man: len(matches[man]) for man in game.men},
            'men_swipes': {man: len(game.swipes[man]) for man in game.men},
            'men_attractiveness': {man: float(traits[man][0]) for man in game.men}
        },
        'Women': {
            'women_matches': {woman: len(matches[woman]) for woman in game.women},
            'women_swipes': {woman: len(game.swipes[woman]) for woman in game.women},
            'women_attractiveness': {woman: float(traits[woman][0]) for woman in game.women}
        }
    }

def run_simulation(config_data):
    """Run the randomized prioritized dating simulation with the given configuration."""
    config = config_data["config"]
    name = config_data["name"]
    
    ppo_type = "mc_ppo" if USE_MONTE_CARLO else "ppo"
    log_dir = f"runs/randomized_{name}_{ppo_type}"
    writer = setup_logging(log_dir)
    
    num_episodes = config.get("num_days", 100)
    attributes = config["attributes"]
    attribute_names = list(attributes.keys())

    reward_function = CoefficientReward()
    environment = env(config, reward_function)
    environment.reset()

    # Setup agents
    input_dim = len(attributes) + 1 
    print(f"\nInitializing Randomized Prioritized Dating Simulation - {name}")
    print(f"Using {'Monte Carlo ' if USE_MONTE_CARLO else ''}PPO")
    print(f"Number of features: {input_dim}")
    print("Features included:")
    for feature in attribute_names:
        print(f"- {feature}")
    print("- remaining_swipes")
    print(f"Reward Function: {reward_function}")
    print(f"Window Range: {config.get('min_window', 5)}-{config.get('max_window', 10)}")
    print(f"Buffer Range: {config.get('min_buffer', 2)}-{config.get('max_buffer', 4)}")

    # Create agents using selected PPO implementation
    create_agents_func = create_mc_ppo_agents if USE_MONTE_CARLO else create_ppo_agents
    agents = create_agents_func(
        environment.game.get_players(),
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=2,
        writer=writer,
        learning_rate=1e-3
    )

    # Store agent attractiveness for visualization
    for agent_id, agent in agents.items():
        agent.attractiveness = environment.game.trait_map[agent_id][0]

    print("\nStarting simulation...")

    # Initialize analytics dictionary
    results = {
        "men_traits": {man: environment.game.get_all_traits()[man].tolist() for man in environment.game.men},
        "women_traits": {woman: environment.game.get_all_traits()[woman].tolist() for woman in environment.game.women},
        "episodes": {}
    }

    # Main training loop
    for episode in range(1, num_episodes + 1):
        environment.reset()
        game = environment.game
        total_reward = 0

        # Run episode
        for agent in environment.agent_iter():
            observation, reward, termination, truncation, info = environment.last()
            total_reward += reward

            if termination or truncation:
                action = None
            elif game.dones[agent]:
                action = 2
            else:
                max_swipes = game.men_max_swipes if agent in game.men else game.women_max_swipes
                remaining_swipes = max_swipes - len(game.swipes[agent])
                observation = np.append(observation, remaining_swipes / max_swipes)

                ppo_agent = agents[agent]
                action, log_prob = ppo_agent.act(observation)
                ppo_agent.remember(observation, action, log_prob.item(), reward, termination or truncation)

            environment.step(action)

        # Calculate rewards and update agents
        rewards = reward_function.calculate_reward(game)
        for agent in agents:
            total_reward += rewards[agent]
            ppo_agent = agents[agent]
            if len(ppo_agent.memory) > 0:
                last_memory = ppo_agent.memory[-1]
                ppo_agent.memory[-1] = (
                    last_memory[0],
                    last_memory[1],
                    last_memory[2],
                    last_memory[3] + rewards[agent],
                    True
                )

        # Update PPO agents
        for agent_id, ppo_agent in agents.items():
            if USE_MONTE_CARLO:
                ppo_agent.monte_carlo_update(episode=episode)
            else:
                ppo_agent.update(episode)

        # Store analytics
        results['episodes'][episode] = collect_episode_metrics(game, agents, total_reward)

        # Print progress
        if episode % 10 == 0:
            print(f"\nEpisode {episode}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average Reward per Agent: {total_reward / len(agents):.3f}")
            print(f"Total Matches: {sum(len(matches) for matches in game.get_all_matches().values()) // 2}")
            print(f"Match/Swipe Ratio: {results['episodes'][episode]['Simulation']['match_swipe_ratio']:.3f}")

    environment.close()
    writer.close()

    # Save analytics and create visualizations
    output_file = save_analytics(results, f"{name}_{ppo_type}_config.json", log_dir)
    create_all_visualizations(results, f'{log_dir}/plots')

    print(f"\n{name} Simulation completed successfully!")
    print(f"Results saved to: {output_file}")
    print(f"Visualizations saved in: {log_dir}/plots")
    print(f"To view training progress, run: tensorboard --logdir={log_dir}")
    
    return agents, results

if __name__ == "__main__":
    config_path = "game_config/multi_sim_config.json"
    configs = load_configs(config_path)
    
    for config_data in configs:
        print(f"\nStarting {config_data['name']} configuration...")
        run_simulation(config_data)
        print(f"Completed {config_data['name']} configuration")