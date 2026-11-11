import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_samples(samples, title="Samples Visualization", save_path=None):
    """
    Visualizes the generated samples with scatter plot and marginal distributions.
    
    Args:
        samples: Generated samples tensor
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    # Convert tensor to numpy for plotting
    samples_np = samples.cpu().numpy()
    
    # Calculate statistics
    sample_mean = torch.mean(samples, dim=0)
    sample_cov = torch.cov(samples.T)
    
    # Move tensors to CPU before converting to numpy
    sample_mean_cpu = sample_mean.cpu()
    sample_cov_cpu = sample_cov.cpu()
    
    # Create the plot
    if samples.shape[1] == 2:  # 2D case
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(title)
        
        # Main scatter plot
        axes[0, 1].scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.6)
        axes[0, 1].set_title("Samples Distribution")
        axes[0, 1].set_xlabel("Dimension 0")
        axes[0, 1].set_ylabel("Dimension 1")
        
        # Marginal distribution for dimension 0
        axes[0, 0].hist(samples_np[:, 0], bins=50, orientation='horizontal', alpha=0.7)
        axes[0, 0].axhline(y=sample_mean_cpu[0].item(), color='r', linestyle='--', label=f'Mean: {sample_mean_cpu[0].item():.2f}')
        axes[0, 0].set_title("Marginal Distribution (Dim 0)")
        axes[0, 0].legend()
        
        # Marginal distribution for dimension 1
        axes[1, 1].hist(samples_np[:, 1], bins=50, orientation='vertical', alpha=0.7)
        axes[1, 1].axvline(x=sample_mean_cpu[1].item(), color='r', linestyle='--', label=f'Mean: {sample_mean_cpu[1].item():.2f}')
        axes[1, 1].set_title("Marginal Distribution (Dim 1)")
        axes[1, 1].legend()
        
        # Remove the unused subplot
        axes[1, 0].axis('off')
        
        # Add statistics text
        stats_text = f"Mean: [{sample_mean_cpu[0].item():.3f}, {sample_mean_cpu[1].item():.3f}]\n"
        stats_text += f"Covariance Matrix:\n{sample_cov_cpu.numpy()}"
        axes[1, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
    else:  # For higher dimensions, show just basic stats
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(title)
        
        # Show statistics
        sample_std = torch.std(samples, dim=0)
        sample_std_cpu = sample_std.cpu()
        ax.text(0.1, 0.8, f"Mean: {sample_mean_cpu.tolist()}", fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.6, f"Std: {sample_std_cpu.tolist()}", fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.4, f"Shape: {list(samples.shape)}", fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.2, f"Covariance Matrix:\n{sample_cov_cpu.numpy()}", fontsize=10, transform=ax.transAxes)
        ax.axis('off')
    
    # Print statistics
    print(f"Mean of generated samples: {sample_mean_cpu}")
    print(f"Covariance matrix of generated samples:\n{sample_cov_cpu}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()