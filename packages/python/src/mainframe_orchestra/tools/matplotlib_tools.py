# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

from typing import List, Union, Any
from ..utils.braintrust_utils import traced

def check_matplotlib_dependencies():
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        return np, plt, mdates
    except ImportError:
        raise ImportError("Matplotlib dependencies are not installed. To use MatplotlibTools, install the required packages with 'pip install matplotlib'")

class MatplotlibTools:
    @traced(type="tool")
    @staticmethod
    def _check_dependencies():
        if 'np' not in globals() or 'plt' not in globals():
            raise ImportError("Matplotlib is not installed. To use MatplotlibTools, install the required packages with 'pip install matplotlib'")

    @traced(type="tool")
    @staticmethod
    def create_line_plot(x: List[List[Union[float, str]]], y: List[List[float]], title: str = None, xlabel: str = "X", ylabel: str = "Y",
                         output_file: str = "line_plot.png") -> Union[Any, str]:
        """
        Create a line plot using the provided x and y data.

        Args:
            x (List[List[Union[float, str]]]): The x-coordinates of the data points for each line.
            y (List[List[float]]): The y-coordinates of the data points for each line.
            title (str, optional): The title of the plot. Defaults to None.
            xlabel (str, optional): The label for the x-axis. Defaults to "X".
            ylabel (str, optional): The label for the y-axis. Defaults to "Y".
            output_file (str, optional): The output file name. Defaults to "line_plot.png".

        Returns:
            Union[Any, str]: The matplotlib figure object, or an error message as a string.
        """
        fig = None
        try:
            np, plt, mdates = check_matplotlib_dependencies()
            if len(x) != len(y):
                raise ValueError(f"The number of x and y lists must be equal. Got {len(x)} x-lists and {len(y)} y-lists. Check your data and try again.")

            for i, (xi, yi) in enumerate(zip(x, y)):
                if not isinstance(xi, list) or not isinstance(yi, list):
                    raise TypeError(f"Both x[{i}] and y[{i}] must be lists. Check your data and try again.")
                if len(xi) != len(yi):
                    raise ValueError(f"The lengths of x[{i}] and y[{i}] must be equal. Got lengths {len(xi)} and {len(yi)}. Check your data and try again.")
                if not all(isinstance(val, (int, float, str)) for val in xi):
                    raise TypeError(f"All values in x[{i}] must be numbers or strings. Check your data and try again.")
                if not all(isinstance(val, (int, float)) for val in yi):
                    raise TypeError(f"All values in y[{i}] must be numbers. Check your data and try again.")

            fig, ax = plt.subplots(figsize=(10, 6))
            for xi, yi in zip(x, y):
                # Convert dates to numerical values
                if all(isinstance(val, str) for val in xi):
                    xi = [mdates.datestr2num(val) for val in xi]
                ax.plot(xi, yi)

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

            fig.tight_layout()
            plt.savefig(output_file)
            return fig

        except ImportError as e:
            return str(e)
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred while creating the line plot: {str(e)}"
            print(error_msg)
            return error_msg
        finally:
            if fig is not None:
                plt.close(fig)  # Ensure the figure is closed to free up memory

    @traced(type="tool")
    @staticmethod
    def create_scatter_plot(x: List[float], y: List[float], title: str = None, xlabel: str = None, ylabel: str = None) -> Union[str, None]:
        """
        Create a scatter plot using the provided x and y data.

        Args:
            x (List[float]): The x-coordinates of the data points.
            y (List[float]): The y-coordinates of the data points.
            title (str, optional): The title of the plot. Defaults to None.
            xlabel (str, optional): The label for the x-axis. Defaults to None.
            ylabel (str, optional): The label for the y-axis. Defaults to None.

        Returns:
            Union[str, None]: The path to the saved plot image file, or an error message as a string.
        """
        try:
            np, plt, _ = check_matplotlib_dependencies()
            if len(x) != len(y):
                raise ValueError("The lengths of x and y must be equal.")

            plt.figure(figsize=(8, 6))
            plt.scatter(x, y)

            if title:
                plt.title(title)
            if xlabel:
                plt.xlabel(xlabel)
            if ylabel:
                plt.ylabel(ylabel)

            plt.tight_layout()
            plot_path = "scatter_plot.png"
            plt.savefig(plot_path)
            plt.close()

            return plot_path

        except ImportError as e:
            return str(e)

        except ValueError as e:
            error_msg = f"Error: {str(e)} Please ensure x and y have the same length."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred while creating the scatter plot: {str(e)}"
            print(error_msg)
            return error_msg

    @traced(type="tool")
    @staticmethod
    def create_bar_plot(x: List[str], y: List[float], title: str = None, xlabel: str = None, ylabel: str = None) -> Union[str, None]:
        """
        Create a bar plot using the provided x and y data.

        Args:
            x (List[str]): The categories for the x-axis.
            y (List[float]): The values for each category.
            title (str, optional): The title of the plot. Defaults to None.
            xlabel (str, optional): The label for the x-axis. Defaults to None.
            ylabel (str, optional): The label for the y-axis. Defaults to None.

        Returns:
            Union[str, None]: The path to the saved plot image file, or an error message as a string.
        """
        try:
            np, plt, _ = check_matplotlib_dependencies()
            if len(x) != len(y):
                raise ValueError("The lengths of x and y must be equal.")

            plt.figure(figsize=(8, 6))
            plt.bar(x, y)

            if title:
                plt.title(title)
            if xlabel:
                plt.xlabel(xlabel)
            if ylabel:
                plt.ylabel(ylabel)

            plt.tight_layout()
            plot_path = "bar_plot.png"
            plt.savefig(plot_path)
            plt.close()

            return plot_path
        except ImportError as e:
            return str(e)

        except ValueError as e:
            error_msg = f"Error: {str(e)} Please ensure x and y have the same length."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred while creating the bar plot: {str(e)}"
            print(error_msg)
            return error_msg

    @traced(type="tool")
    @staticmethod
    def create_histogram(data: List[float], bins: int = 10, title: str = None, xlabel: str = None, ylabel: str = None) -> Union[str, None]:
        """
        Create a histogram using the provided data.

        Args:
            data (List[float]): The data to plot in the histogram.
            bins (int, optional): The number of bins for the histogram. Defaults to 10.
            title (str, optional): The title of the plot. Defaults to None.
            xlabel (str, optional): The label for the x-axis. Defaults to None.
            ylabel (str, optional): The label for the y-axis. Defaults to None.

        Returns:
            Union[str, None]: The path to the saved plot image file, or an error message as a string.
        """
        try:
            np, plt, _ = check_matplotlib_dependencies()
            plt.figure(figsize=(8, 6))
            plt.hist(data, bins=bins)

            if title:
                plt.title(title)
            if xlabel:
                plt.xlabel(xlabel)
            if ylabel:
                plt.ylabel(ylabel)

            plt.tight_layout()
            plot_path = "histogram.png"
            plt.savefig(plot_path)
            plt.close()

            return plot_path

        except ImportError as e:
            return str(e)
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred while creating the histogram: {str(e)}"
            print(error_msg)
            return error_msg

    @traced(type="tool")
    @staticmethod
    def create_heatmap(data: List[List[float]], title: str = None, xlabel: str = None, ylabel: str = None) -> Union[str, None]:
        """
        Create a heatmap using the provided 2D data.

        Args:
            data (List[List[float]]): The 2D data to plot in the heatmap.
            title (str, optional): The title of the plot. Defaults to None.
            xlabel (str, optional): The label for the x-axis. Defaults to None.
            ylabel (str, optional): The label for the y-axis. Defaults to None.

        Returns:
            Union[str, None]: The path to the saved plot image file, or an error message as a string.
        """
        try:
            np, plt, _ = check_matplotlib_dependencies()
            data = np.array(data)

            plt.figure(figsize=(8, 6))
            plt.imshow(data, cmap='viridis')
            plt.colorbar()

            if title:
                plt.title(title)
            if xlabel:
                plt.xlabel(xlabel)
            if ylabel:
                plt.ylabel(ylabel)

            plt.tight_layout()
            plot_path = "heatmap.png"
            plt.savefig(plot_path)
            plt.close()

            return plot_path

        except ImportError as e:
            return str(e)
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred while creating the heatmap: {str(e)}"
            print(error_msg)
            return error_msg