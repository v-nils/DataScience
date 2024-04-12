import os
import matplotlib.pyplot as plt


def process_plot(plot: plt, save_path: str | None = None) -> None:
    """
    Save the plot to the specified path, or display it, based on the value of save_path.

    :param plot: matplotlib.pyplot object
    :param save_path: Path to save the plot. If None, the plot is displayed.
    """
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            raise FileNotFoundError(f"Path {save_path} does not exist")

        print('Save figure to:', save_path)
        plot.savefig(save_path, bbox_inches='tight')
    else:
        plot.show()
