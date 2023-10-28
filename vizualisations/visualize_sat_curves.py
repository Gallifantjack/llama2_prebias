import polars as pl
import matplotlib.pyplot as plt


def normalize_data(data_list):
    """Normalize data to [0, 1] range."""
    min_val = min(data_list)
    max_val = max(data_list)

    if min_val == max_val:
        return [0] * len(data_list)

    return [(x - min_val) / (max_val - min_val) for x in data_list]


def extract_checkpoint_number(filename):
    try:
        # Updated extraction based on the 'ckpt_NUMBER.pt' format
        return int(filename.split("ckpt_")[1].split(".pt")[0])
    except IndexError:
        print(f"Unexpected filename structure: {filename}")
        return None


def plot_metrics_from_parquet(ax, file_path, name):
    # Load the Parquet file into a Polars DataFrame
    df = pl.read_parquet(file_path)

    # Sort the DataFrame by checkpoint number for a consistent progression in the plots
    df = df.sort("checkpoint_name")

    checkpoints = df["checkpoint_name"].to_list()

    excluded_columns = ["checkpoint_name", "text", "global_idx"]

    # Get the list of columns to plot
    metrics = [col for col in df.columns if col not in excluded_columns]

    # For each metric, normalize and plot
    for metric in metrics:
        normalized_values = normalize_data(df[metric].to_list())
        ax.plot(checkpoints, normalized_values, label=metric)

    # Configure the plot
    ax.set_title(f"Normalized {name} Metrics over Checkpoints")
    ax.set_xlabel("Checkpoint Number (Epochs)")
    ax.set_ylabel("Normalized Score vs Max Score")
    ax.legend(loc="upper left")
    ax.grid(True)


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # plot for output benchmarks
    output_csv_file_path = "out/tables/summary.csv"
    plot_metrics_from_parquet(ax1, output_csv_file_path, "sat_curves_output")

    # plot for batch benchmarks
    batch_csv_file_path = "out/tables/batch_results.csv"
    plot_metrics_from_parquet(ax2, batch_csv_file_path, "sat_curves_batch")

    plt.tight_layout()
    plt.savefig("out/visualize/sat/batch_output_curves.png")
    plt.close()
