import argparse
from textwrap import wrap

import matplotlib
# Using matplotlib to save to file, not to show(). No backend needed.
matplotlib.use('Agg')  # noqa E402 to avoid complaining about imports below
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


class EspPlotter:

    @staticmethod
    def plot_stats(filename, title, metric_name="score", solved=None):
        with open(filename) as csv_file:
            df = pd.read_csv(csv_file, sep=',')
            columns_to_plot = ["max_" + metric_name,
                               "min_" + metric_name,
                               "mean_" + metric_name]

            # Append a constant target if needed
            if solved:
                df["solved"] = solved
                columns_to_plot.append("solved")

            # Plot
            df.plot(x="generation",
                    y=columns_to_plot,
                    kind="line")
            ax = plt.gca()

            # Title
            ax.set_title("\n".join(wrap(title, 60)))

            # Force integers for the x axis (generation)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            fig = ax.get_figure()

            # Save the figure to a png file, in the same folder
            output_filename = filename[:-4] + ".png"
            fig.savefig(output_filename)

            # Close the figure so it's not displayed in 'inline' mode.
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        type=str,
                        help='Path to the file containing the experiment stats')
    parser.add_argument("title",
                        type=str,
                        help='Title of the plot')
    parser.add_argument("-m", "--metric", default="score",
                        dest="metric",
                        type=str,
                        help="Name of the metric to plot")
    parser.add_argument("-s", "--solved", default=None,
                        dest="solved",
                        type=float,
                        help='Fitness at which the problem is considered solved')
    args = parser.parse_args()
    EspPlotter().plot_stats(args.filename, args.title, args.metric, args.solved)
