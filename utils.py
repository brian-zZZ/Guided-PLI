import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def plot_train_dev_metric(epochs, train_metric, eval_metric, base_path, metric_name, dataset_name):
    plt.plot(epochs, train_metric, '#3fc1fd', label='Train')
    plt.plot(epochs, eval_metric, '#d09fff', label='Validation')
    # plt.plot([330, 330], [0.9773016059994697-0.1, 1.0378530149936677+0.1], '#fd8989', label='Take the model parameters of the epoch')
    plt.title('Train and Validation {} on {}'.format(metric_name, dataset_name))
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(os.path.join(base_path, dataset_name + '_' + metric_name +'.jpg'))
    plt.cla()


def set_seed(seed):
    np.random.seed(seed) # fix random seed to reproduce results
    torch.manual_seed(seed)         # Current CPU
    torch.cuda.manual_seed(seed)    # Current GPU
