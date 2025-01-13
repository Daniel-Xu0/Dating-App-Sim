import os
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_last_episode(data):
    """Helper function to get the last episode number regardless of type"""
    episodes = list(data['episodes'].keys())
    return max(int(ep) if isinstance(ep, str) else ep for ep in episodes)

def preprocess_data(data, gender, group, episode=-1):
    """Preprocess data for specific gender and metric"""
    if episode == -1:
        episode_key = get_last_episode(data)
    else:
        episode_key = episode

    # Extract matches and traits
    matches = data["episodes"][episode_key][gender][group]
    traits = data[f"{gender.lower()}_traits"]

    records = []
    for agent_id in traits.keys():
        if agent_id in matches:
            matches_count = len(matches[agent_id]) if isinstance(matches[agent_id], (list, set)) else matches[agent_id]
            agent_trait = traits[agent_id][0]
            records.append((agent_trait, matches_count))

    return pd.DataFrame(records, columns=["Agent Attractiveness", "Number of Matches"])

def create_evaluation_metrics(data, output_dir):
    """Plot evaluation metrics between men & women"""
    last_episode = get_last_episode(data)
    episode_data = data['episodes'][last_episode]
    
    men_metrics = episode_data['Men']
    women_metrics = episode_data['Women']
    
    men_matches = men_metrics['men_matches']
    women_matches = women_metrics['women_matches']
    
    men_total_swipes = sum(men_metrics['men_swipes'].values())
    women_total_swipes = sum(women_metrics['women_swipes'].values())
    men_total_matches = len(men_matches)
    women_total_matches = len(women_matches)
    
    metrics = {
        "Number of Matches": [men_total_matches, women_total_matches],
        "Match Rate": [
            men_total_matches / men_total_swipes if men_total_swipes > 0 else 0,
            women_total_matches / women_total_swipes if women_total_swipes > 0 else 0
        ],
        "Total Swipes": [men_total_swipes, women_total_swipes]
    }

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(metrics.keys()))

    for i, (metric_name, values) in enumerate(metrics.items(), start=1):
        fig.add_trace(
            go.Bar(name="Men", x=["Men"], y=[values[0]], marker_color="blue", showlegend=(i == 1)),
            row=1, col=i
        )
        fig.add_trace(
            go.Bar(name="Women", x=["Women"], y=[values[1]], marker_color="pink", showlegend=(i == 1)),
            row=1, col=i
        )

    fig.update_layout(
        title_text="Evaluation Metrics by Gender",
        title_x=0.5,
        showlegend=True,
        width=1200,
        height=400
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_metrics.html")
    fig.write_html(output_path)

def create_time_series_plots(data, output_dir):
    """Create time series plots of simulation metrics"""
    episodes = sorted(int(ep) if isinstance(ep, str) else ep for ep in data['episodes'].keys())
    
    metrics = {
        'Total Matches': [data['episodes'][ep]['Simulation']['total_matches'] for ep in episodes],
        'Average Reward': [data['episodes'][ep]['Simulation']['avg_reward'] for ep in episodes],
        'Match/Swipe Ratio': [data['episodes'][ep]['Simulation']['match_swipe_ratio'] for ep in episodes]
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(metrics.keys()) + [""],
        specs=[[{}, {}], [{}, None]]
    )

    positions = {
        'Total Matches': (1, 1),
        'Average Reward': (1, 2),
        'Match/Swipe Ratio': (2, 1)
    }

    for metric, values in metrics.items():
        row, col = positions[metric]
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=values,
                mode='lines+markers',
                name=metric,
                line=dict(width=2),
                marker=dict(size=6)
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Episode", row=row, col=col)
        fig.update_yaxes(title_text=metric, row=row, col=col)

    fig.update_layout(
        height=800,
        width=1200,
        title={
            'text': "Dating Simulation Results",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'simulation_results.html')
    fig.write_html(output_path)

def create_attractiveness_reward_plot(results, output_dir):
    """Create scatter plot of attractiveness vs matches with trend lines"""
    last_episode = get_last_episode(results)
    final_data = results['episodes'][last_episode]
    
    men_data = {
        'attractiveness': [traits[0] for traits in results['men_traits'].values()],
        'matches': [final_data['Men']['men_matches'][man_id] for man_id in results['men_traits'].keys()],
        'swipes': [final_data['Men']['men_swipes'][man_id] for man_id in results['men_traits'].keys()]
    }
    
    women_data = {
        'attractiveness': [traits[0] for traits in results['women_traits'].values()],
        'matches': [final_data['Women']['women_matches'][woman_id] for woman_id in results['women_traits'].keys()],
        'swipes': [final_data['Women']['women_swipes'][woman_id] for woman_id in results['women_traits'].keys()]
    }
    
    fig = go.Figure()
    
    for gender, data, color in [('Men', men_data, 'blue'), ('Women', women_data, 'red')]:
        fig.add_trace(go.Scatter(
            x=data['attractiveness'],
            y=data['matches'],
            mode='markers',
            name=gender,
            marker=dict(color=color, size=10, opacity=0.6),
            hovertemplate=(
                f'<b>{gender[:-1]}</b><br>' +
                'Attractiveness: %{x:.2f}<br>' +
                'Matches: %{y}<br>' +
                'Swipes: %{customdata}<br>' +
                '<extra></extra>'
            ),
            customdata=data['swipes']
        ))
        
        if len(data['attractiveness']) > 1:
            z = np.polyfit(data['attractiveness'], data['matches'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(data['attractiveness']), max(data['attractiveness']), 100)
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name=f'{gender} Trend',
                line=dict(dash='dash', width=1),
                opacity=0.5
            ))
    
    fig.update_layout(
        title={
            'text': 'Agent Attractiveness vs Number of Matches',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Attractiveness Score',
        yaxis_title='Number of Matches',
        width=1000,
        height=800,
        hovermode='closest',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    fig.update_xaxes(range=[0, 1], gridcolor='lightgray', gridwidth=0.5, showgrid=True)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5, showgrid=True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'attractiveness_vs_matches.html')
    fig.write_html(output_path)

def create_attractiveness_heatmap(data, num_bins=10, output_dir='visualizations'):
    """Create heatmap showing match frequency by attractiveness levels"""
    men_df = preprocess_data(data, "Men", "men_matches")
    women_df = preprocess_data(data, "Women", "women_matches")

    attr_bins = np.linspace(0, 1, num_bins + 1)
    max_matches = max(
        men_df["Number of Matches"].max(),
        women_df["Number of Matches"].max()
    )
    match_bins = np.linspace(0, max_matches, num_bins + 1)

    men_df["Agent Bin"] = pd.cut(men_df["Agent Attractiveness"], bins=attr_bins, labels=attr_bins[:-1])
    men_df["Match Bin"] = pd.cut(men_df["Number of Matches"], bins=match_bins, labels=match_bins[:-1])
    women_df["Agent Bin"] = pd.cut(women_df["Agent Attractiveness"], bins=attr_bins, labels=attr_bins[:-1])
    women_df["Match Bin"] = pd.cut(women_df["Number of Matches"], bins=match_bins, labels=match_bins[:-1])

    men_heatmap = men_df.groupby(["Agent Bin", "Match Bin"], observed=True).size().unstack(fill_value=0)
    women_heatmap = women_df.groupby(["Agent Bin", "Match Bin"], observed=True).size().unstack(fill_value=0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Men: Attractiveness vs Number of Matches", "Women: Attractiveness vs Number of Matches"),
        shared_yaxes=True,
        x_title="Number of Matches",
        y_title="Agent Attractiveness"
    )

    fig.add_trace(
        go.Heatmap(
            z=men_heatmap.values,
            x=men_heatmap.columns.astype(float),
            y=men_heatmap.index.astype(float),
            colorscale='Blues',
            colorbar=dict(title="Frequency"),
            hovertemplate='Agent Attractiveness: %{y}<br>Average Match Attractiveness: %{x}<br>Frequency: %{z}<extra></extra>',
            showscale=False,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=women_heatmap.values,
            x=women_heatmap.columns.astype(float),
            y=women_heatmap.index.astype(float),
            colorscale='RdPu',
            colorbar=dict(title="Frequency"),
            hovertemplate='Agent Attractiveness: %{y}<br>Average Match Attractiveness: %{x}<br>Frequency: %{z}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="Match Frequency Heatmap: Men vs. Women",
        showlegend=False,
        width=1200,
        height=600
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'attractiveness_heatmap.html')
    fig.write_html(output_path)

def create_metrics_animation(data, output_dir='visualizations'):
    """Create animated visualization of metrics over time"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = sorted(int(ep) if isinstance(ep, str) else ep for ep in data['episodes'].keys())
    metrics = {
        'total_matches': [data['episodes'][ep]['Simulation']['total_matches'] for ep in episodes],
        'avg_reward': [data['episodes'][ep]['Simulation']['avg_reward'] for ep in episodes],
        'match_ratio': [data['episodes'][ep]['Simulation']['match_swipe_ratio'] for ep in episodes]
    }
    
    lines = []
    for key in metrics:
        line, = ax.plot([], [], label=key.replace('_', ' ').title())
        lines.append((key, line))
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.set_title('Dating Simulation Metrics Over Time')
    ax.legend()
    
    def init():
        for _, line in lines:
            line.set_data([], [])
        return [line for _, line in lines]
    
    def animate(frame):
        frame = min(frame + 1, len(episodes))
        for key, line in lines:
            line.set_data(episodes[:frame], metrics[key][:frame])
        ax.relim()
        ax.autoscale_view()
        return [line for _, line in lines]
    
    anim = FuncAnimation(
        fig, animate,
        init_func=init,
        frames=len(episodes),
        interval=100,
        blit=True,
        repeat=False
    )
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics_animation.gif')
    anim.save(output_path, writer='pillow')
    plt.close()

def create_all_visualizations(results, output_dir):
    """Create all visualizations for the simulation results"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreating visualizations in {output_dir}...")
    
    try:
        print("Creating evaluation metrics plot...")
        create_evaluation_metrics(results, output_dir)
        
        print("Creating time series plots...")
        create_time_series_plots(results, output_dir)
        
        print("Creating attractiveness vs matches plot...")
        create_attractiveness_reward_plot(results, output_dir)
        
        print("Creating attractiveness heatmap...")
        create_attractiveness_heatmap(results, output_dir=output_dir)
        
        print("Creating metrics animation...")
        create_metrics_animation(results, output_dir=output_dir)
        
        print("\nAll visualizations have been created successfully!")
        print("\nGenerated files:")
        print("- evaluation_metrics.html")
        print("- simulation_results.html")
        print("- attractiveness_vs_matches.html")
        print("- attractiveness_heatmap.html")
        print("- metrics_animation.gif")
        
    except Exception as e:
        print(f"\nError creating visualizations: {str(e)}")
        print("Full error:")
        import traceback
        traceback.print_exc()
        raise