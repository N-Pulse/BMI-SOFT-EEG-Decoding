from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def show_results(pred_labels, true_labels=None, metrics=["accuracy", "recall", "precision", "f1-score"], labels=None):
    """
    Show the results
    """
    if true_labels is not None:
        # Offline results
        print_metrics(true_labels, pred_labels, metrics)
        plot_confusion_matrix(true_labels, pred_labels, labels)
        
    plot_labels_series(pred_labels, true_labels=true_labels)

def print_metrics(true_labels, pred_labels, metrics):
    """
    Computes and plots the specified metrics.

    Parameters:
    true_labels (array-like): Ground truth labels.
    pred_labels (array-like): Predicted labels.
    metrics (list): List of metric names (as strings) to compute and display.
                    Supported metrics: 'accuracy', 'recall', 'precision', 'f1-score'.
    """
    metric_values = {}

    for metric in metrics:
        if metric == 'accuracy':
            metric_values['accuracy'] = accuracy_score(true_labels, pred_labels)
        elif metric == 'recall':
            metric_values['recall'] = recall_score(true_labels, pred_labels, average='binary')
        elif metric == 'precision':
            metric_values['precision'] = precision_score(true_labels, pred_labels, average='binary')
        elif metric == 'f1-score':
            metric_values['f1-score'] = f1_score(true_labels, pred_labels, average='binary')
        else:
            print(f"Warning: Metric '{metric}' is not supported and will be skipped.")

    # Print the metrics
    for metric, value in metric_values.items():
        print(f"{metric.capitalize()}: {value:.4f}")


def plot_confusion_matrix(true_labels, pred_labels, labels):
    confmat = confusion_matrix(true_labels, pred_labels, labels, normalize='true')
    plt.figure(figsize=(10, 7))
    sns.heatmap(confmat, annot=True, fmt=".2f", cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.show()


def plot_labels_series(pred_labels, true_labels=None):
    plt.figure(figsize=(12, 6))  # Adjust the figure size
    if true_labels is not None:
        plt.plot(true_labels, label='Actual Labels', color='blue')
    plt.plot(pred_labels, label='Predicted Labels', color='red', alpha=0.5)
    plt.title('Predicted vs Actual Labels')
    plt.xlabel('Time (s)')
    plt.ylabel('Gesture')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    