import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib import rcParams

# Set publication-quality defaults
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['axes.linewidth'] = 1.5

def plot_confusion_matrix(cm, ax, title=None):
    """Plot a publication-quality confusion matrix."""
    # Calculate percentages for displaying alongside raw counts
    row_sums = np.sum(cm, axis=1, keepdims=True)
    cm_percentages = cm / row_sums * 100
    
    # Create annotations with both count and percentage
    annot = np.array([[f"{val}\n({pct:.1f}%)" for val, pct in zip(row, row_pct)] 
                      for row, row_pct in zip(cm, cm_percentages)])
    
    # Plot with optimized settings for paper readability
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"],
                annot_kws={"size": 32, "weight": "bold"},
                cbar=False)
    
    # Bold, clean labels
    ax.set_xlabel('Predicted', fontsize=22, fontweight='bold')
    ax.set_ylabel('True', fontsize=22, fontweight='bold')
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=30, fontweight='bold', pad=10)
    
    # Ensure tick labels are readable
    ax.tick_params(axis='both', which='major', labelsize=24, width=1.5)

def create_direct_confusion_matrix_grid(confusion_matrices, output_path='direct_confusion_matrix_grid.png'):
    """Create publication-ready confusion matrix grid."""
    # Create figure with high-quality settings
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    
    # Define row titles (metrics)
    row_titles = ["Mistake Identification", "Actionability"]
    
    # Define column titles (models)
    model_titles = ["Encoder\n(BERT)", "Encoder-Decoder\n(T5)", "Decoder\n(Llama)"]
    
    # Plot confusion matrices
    for i, row_title in enumerate(row_titles):
        for j in range(3):
            plot_confusion_matrix(confusion_matrices[i*3+j], axs[i, j])
    
    # Add row titles on the left side
    for i, title in enumerate(row_titles):
        fig.text(0.01, 0.75 - i*0.5, title, fontsize=28, rotation=90, 
                 ha='center', va='center', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, pad=3, edgecolor='gainsboro'))
    
    # Titles
    for j, title in enumerate(model_titles):
        x_position = 0.18 + j * 0.32
        fig.text(x_position, 0.95, title, fontsize=30, ha='center', va='center', 
                 fontweight='bold', linespacing=1.2,
                 bbox=dict(facecolor='white', alpha=0.8, pad=3, edgecolor='gainsboro'))
    
    # Optimize spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.07, right=0.95, top=0.9, bottom=0.08)
    
    # Save in multiple formats
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight')
    
    # plt.show()
    print(f"Confusion matrix grid saved in PNG and SVG formats")

# Example usage
if __name__ == "__main__":
    # Define confusion matrices for each model and metric
    cm_mistake_identification_encoder = np.array([[51, 31], [9, 405]])
    cm_mistake_identification_encoder_decoder = np.array([[38, 36], [12, 410]])
    cm_mistake_identification_decoder = np.array([[420, 0], [0, 420]])
    
    cm_actionability_encoder = np.array([[111, 52], [29, 304]])
    cm_actionability_encoder_decoder = np.array([[86, 74], [41, 295]])
    cm_actionability_decoder = np.array([[420, 0], [0, 420]])
    
    confusion_matrices = [
        cm_mistake_identification_encoder,
        cm_mistake_identification_encoder_decoder,
        cm_mistake_identification_decoder,
        cm_actionability_encoder,
        cm_actionability_encoder_decoder,
        cm_actionability_decoder
    ]
    
    # Create the grid of confusion matrices
    create_direct_confusion_matrix_grid(confusion_matrices)