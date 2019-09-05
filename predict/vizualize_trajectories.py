import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

_LINE_COLORS = list(colors.BASE_COLORS.keys())


def _get_line_color(i):
    return _LINE_COLORS[i % len(_LINE_COLORS)]


def visualize_trajectories(y_true_s, y_pred_s, obs_len, pred_len):
    """Vizualize trajectories in the sample s.

    :param y_true_s: (obs_len + pred_len, max_n_peds, 3)
    :param y_pred_s: (obs_len + pred_len, max_n_peds, 3)
    :return:
    """
    assert y_true_s.shape == y_pred_s.shape
    assert y_true_s.shape[1] == 20

    # not_exist_pid = 0
    # exist_peds = y_true_s[0, :, 0] != not_exist_pid
    # exist_pids = y_true_s[0, exist_peds, 0].astype(np.int32)
    # pos_true_s = y_true_s[:, exist_peds, 1:]
    # pos_pred_s = y_pred_s[:, exist_peds, 1:]

    fig, ax = plt.subplots()
    # n_exist_pids = len(exist_pids)
    n_exist_pids = y_true_s.shape[1]
    # for i in range(n_exist_pids):
    #     x_pos_true, y_pos_true = y_true_s[i,:, :][:,0], y_true_s[i,:,:][:,1]
    #     x_pos_true, y_pos_true = x_pos_true.ravel(), y_pos_true.ravel()
    #
    #     x_pos_pred, y_pos_pred = y_pred_s[i,:,:][:.0], y_pred_s[i,:,:][:,1]
    #     x_pos_pred, y_pos_pred = x_pos_pred.ravel(), y_pos_pred.ravel()
    #
    #     line_color = _get_line_color(i)

    x_pos_true, y_pos_true = y_true_s[0,:, :][:,0], y_true_s[0,:,:][:,1]
    x_pos_true, y_pos_true = x_pos_true.ravel(), y_pos_true.ravel()

    x_pos_pred, y_pos_pred = y_pred_s[0,:,:][:,0], y_pred_s[0,:,:][:,1]
    x_pos_pred, y_pos_pred = x_pos_pred.ravel(), y_pos_pred.ravel()

    line_color = _get_line_color(2)
# 画图
    ax.plot(x_pos_pred, y_pos_pred, line_color, linestyle=":")
    ax.plot(x_pos_true, y_pos_true, line_color)
    ax.scatter(x_pos_pred, y_pos_pred, c=line_color, marker=".")
    ax.scatter(x_pos_true, x_pos_true, c=line_color)


        # draw line
        # ax.plot(x_pos_pred, y_pos_pred, line_color, linestyle=":")
        # ax.plot(x_pos_true, y_pos_true, line_color)
        # ax.scatter(x_pos_pred, y_pos_pred, c=line_color, marker=".")
        # ax.scatter(x_pos_true, x_pos_true, c=line_color)
    data_path="/home/lianli/zhoubin_work/zhoubin_work/ss-lstm_0715/person.txt"
    data_pos = open(data_path,'w')
    data_pos.write(str(x_pos_pred))
    data_pos.write(str(y_pos_pred))


    # set figure bound
    x_min, x_max = -1.5, 0.5
    y_min, y_max = -1.5, 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return fig
