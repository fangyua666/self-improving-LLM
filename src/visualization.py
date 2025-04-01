# src/visualization.py
import plotly.graph_objects as go
import wandb

def plot_accuracy_improvement(diff_model_performance):
    """
    Plot accuracy improvement over self-improvement rounds.
    
    Args:
        diff_model_performance (dict): Dictionary containing performance data.
        
    Returns:
        go.Figure: Plotly figure.
    """
    fig = go.Figure()

    x_values = [i for i in range(11, 21)]

    i = 0
    for m_performace in diff_model_performance.values():
        fig.add_trace(go.Scatter(x=x_values,
                                y=m_performace,
                                mode='lines+markers',
                                name=f"Self-improvement round {i}"))
        i += 1

    fig.update_layout(title="10 rounds of self-improvement, majority voting", 
                      xaxis_title="number of digits", 
                      yaxis_title="Average Accuracy")
    fig.update_yaxes(range=[-0.02, 1.02])
    fig.update_xaxes(tickmode="array", tickvals=x_values)
    fig.update_layout(width=1000, height=500)
    
    return fig

def plot_wrong_answers_accuracy(evaluation_data):
    """
    Plot accuracy on wrong answers across rounds.
    
    Args:
        evaluation_data (dict): Dictionary with evaluation data.
        
    Returns:
        go.Figure: Plotly figure.
    """
    fig = go.Figure()

    for t, (checkpoints, accuracies) in evaluation_data.items():
        fig.add_trace(go.Scatter(
            x=checkpoints,
            y=accuracies,
            mode='lines+markers',
            name=f"Wrong Answers from Round {t}",
            hovertemplate="Checkpoint: %{x}<br>Accuracy: %{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title="Model Accuracy on Wrong Answers Across Rounds",
        xaxis_title="Model Checkpoint Round",
        yaxis_title="Accuracy on Wrong Answers",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig
def log_wandb_chart(fig, name):
    """
    Log a Plotly figure to W&B.
    
    Args:
        fig (go.Figure): Plotly figure.
        name (str): Name for the chart.
    """
    wandb.log({f"{name}": wandb.Html(fig.to_html())})
