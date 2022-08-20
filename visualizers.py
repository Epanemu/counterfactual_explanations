import matplotlib.pyplot as plt
import numpy as np


def array_of_coefficient_names(self):
    out = list()
    for i in range(self.cox.shape[0]):
        out.append(self.cox[i].name)
        for j in (self.cox[i].unique):
            out.append(self.cox[i].name + ' ' + str(j))

    medians = np.median(self.encoded, 0)
    recalc = (medians == 0)
    medians[recalc] = np.mean(self.encoded, 0)[recalc]
    return np.asarray(out), np.asarray(medians)


def plot_coefficients(exp, filename='out.png', med_weight=False):
    bias = exp.model.intercept_[0]
    coef = exp.model.coef_[0]
    names, med = array_of_coefficient_names(exp)
    if med_weight:
        order = np.argsort(np.abs(coef * med))[::-1]
    else:
        order = np.argsort(np.abs(coef))[::-1]
    y_pos = np.arange(1 + coef.shape[0])
    plt.rcdefaults()
    width = 8
    height = 12

    fig, ax = plt.subplots(figsize=(width, height))
    ax.barh(y_pos, np.hstack((bias, coef[order])), align='center',
            color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.hstack(('Bias', names[order])))
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Magnitude ')
    # ax.set_xscale('symlog')
    ax.set_title('Variable Weights')
    plt.savefig(filename)
    plt.show()
