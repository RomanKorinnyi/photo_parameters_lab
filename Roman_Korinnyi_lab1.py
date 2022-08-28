import collections
import random
import warnings
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
from scipy.stats._continuous_distns import _distn_names


class Image:

    def __init__(self, num):
        self.img_path = str(Path("mirflickr", f"im{num}.jpg"))
        self.img = cv.imread(self.img_path)
        self.b, self.g, self.r = cv.split(self.img)
        self.blue_list = [num for numbers in self.b for num in numbers]
        self.green_list = [num for numbers in self.g for num in numbers]
        self.red_list = [num for numbers in self.r for num in numbers]

        self.distrs = ["norm", "gamma", "laplace"]
        self.show_plots = True

    def min_max(self, chanel):
        """
        Finds the maximum and minimum value for each color channel in the picture
        :return: tuple
        """
        return np.amin(chanel), np.amax(chanel)

    def show_picture(self):
        """
        Shows the picture in each color channel
        :return:
        """
        cv.imshow("Model Blue Image", self.b)
        cv.imshow("Model Green Image", self.g)
        cv.imshow("Model Red Image", self.r)
        cv.waitKey(0)

    def math_expectation(self, list_of_intensities):
        """
        Calculates the math expectation for chanel
        :param list_of_intensities:
        :return:
        """
        c_amounts = dict(collections.Counter(list_of_intensities))
        M = 0
        for number in c_amounts:
            M += (number * c_amounts[number] / len(list_of_intensities))
        return M

    def dispersion(self, list_of_intensities):
        c_amounts = dict(collections.Counter(list_of_intensities))
        M = self.math_expectation(list_of_intensities)
        D = 0
        for number in c_amounts:
            D += (number ** 2 * c_amounts[number] / len(list_of_intensities))
        D -= M ** 2
        return D

    def median(self, list_of_intensities):
        return np.median(list_of_intensities)

    def interquartile_range(self, list_of_intensities):
        return np.percentile(list_of_intensities, 75) - np.percentile(list_of_intensities, 25)

    def asymmetry(self, list_of_intensities):
        """
        The ratio of the difference between the mean and the median to the standard deviation
        :param list_of_intensities: list
        :return: float
        """
        return st.skew(np.array(list_of_intensities))

    def kurtosis(self, list_of_intensities):
        """
        A measure of the sharpness of the peak of a distribution
        :param list_of_intensities: list
        :return:
        """
        return st.kurtosis(np.array(list_of_intensities))

    def make_pdf(self, dist, params, size=10000):
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        return pdf

    def best_fit_distribution(self, data, bins=200, ax=None):
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        best_distributions = []
        for ii, distribution in enumerate([d for d in self.distrs if d in _distn_names]):
            distribution = getattr(st, distribution)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    params = distribution.fit(data)
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax, label=f"{distribution.name}", legend=True)
                    except Exception:
                        pass
                    best_distributions.append((distribution, params, sse))
            except Exception:
                pass

        return sorted(best_distributions, key=lambda x: x[2])[0]

    def build_distributions(self, channel, ch_name, pic_path):
        list_of_intensities = [num for numbers in channel for num in numbers]
        c_amounts = dict(collections.Counter(list_of_intensities))
        hyst = {}
        all_pix = len(list_of_intensities)
        for numm in c_amounts:
            hyst[numm] = c_amounts[numm] / all_pix

        data = pd.Series(channel.ravel())

        plt.figure(figsize=(12, 8))
        ax = data.plot(kind='hist', bins=256, density=True, color="red")
        yLim = ax.get_ylim()
        best_distr = self.best_fit_distribution(data, bins=256, ax=ax)
        pdf = self.make_pdf(best_distr[0], best_distr[1])
        ax = pdf.plot(label="PDF", legend=True)
        ax.set_ylim(yLim)
        ax.set_xlim(0)
        data.plot(kind='hist', bins=256, density=True, label="DATA", legend=True)
        plt.title(f"Pic: {pic_path}, Channel: {ch_name}")

        if self.show_plots:
            plt.show()
        else:
            plt.close()
        dist_name = best_distr[0].name
        return {dist_name}


def all_actions_for_picture(n):
    im = Image(n)
    print(f"{im.img_path}".center(50, "*"))
    for list_of_intensities, channel_array, ch_name in zip(
            [im.red_list, im.blue_list, im.green_list],
            [im.r, im.b, im.g],
            ["Red", "Blue", "Green"]
    ):
        print("-" * 50)
        print(f"Min and max values for channel {ch_name}: {im.min_max(list_of_intensities)}")
        print(f"Math expectation for channel {ch_name}: {im.math_expectation(list_of_intensities)}")
        print(f"Dispersion for channel {ch_name}: {im.dispersion(list_of_intensities)}")
        print(f"Median value for channel {ch_name}: {im.median(list_of_intensities)}")
        print(f"Interquartile range for channel {ch_name}: {im.interquartile_range(list_of_intensities)}")
        print(f"Asymmetry factor {ch_name}: {im.asymmetry(list_of_intensities)}")
        print(f"Kurtosis factor {ch_name}: {im.kurtosis(list_of_intensities)}")
        print(f"Best distribution for channel {ch_name}: {im.build_distributions(channel_array, ch_name, im.img_path)}")
        print("\n")


if __name__ == "__main__":
    random.seed(5)
    for i in range(1):
        n = random.randint(1, 25000)
        all_actions_for_picture(n)
