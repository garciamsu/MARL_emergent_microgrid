import matplotlib.pyplot as plt


def plot_metric(df, field, ylabel, filename_svg):
    plt.figure()
    plt.plot(df["Episode"], df[field], drawstyle="steps-post")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename_svg, format="svg")
    plt.close()
