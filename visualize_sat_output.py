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


def plot_metrics_from_csv(file_path):
    # Load the CSV file into a Polars DataFrame
    df = pl.read_csv(file_path)

    # Extract the checkpoint numbers
    df = df.with_columns(
        [
            df["checkpoint_name"]
            .map_elements(extract_checkpoint_number)
            .alias("checkpoint_number")
        ]
    )

    # Remove rows where checkpoint number extraction failed
    df = df.filter(df["checkpoint_number"].is_not_null())

    # Sort the DataFrame by checkpoint number for a consistent progression in the plots
    df = df.sort("checkpoint_number")

    checkpoints = df["checkpoint_number"].to_list()

    # Exclude certain columns from plotting
    excluded_columns = ["checkpoint_name", "text", "checkpoint_number"]

    # Get the list of columns to plot
    metrics = [col for col in df.columns if col not in excluded_columns]

    # For each metric, normalize and plot
    for metric in metrics:
        normalized_values = normalize_data(df[metric].to_list())
        plt.plot(checkpoints, normalized_values, label=metric)

    # Configure the plot
    plt.title("Normalized Metrics over Checkpoints")
    plt.xlabel("Checkpoint Number (Epochs)")
    plt.ylabel("Normalized Score")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig("out/visualize/sat/sat_curves_output.png")


if __name__ == "__main__":
    csv_file_path = "out/tables/summary.csv"
    plot_metrics_from_csv(csv_file_path)
