import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

class Plot:
    def __init__(self):
        pass

    def showPlot(points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)