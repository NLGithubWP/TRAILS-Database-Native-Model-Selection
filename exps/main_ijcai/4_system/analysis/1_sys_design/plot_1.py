import numpy as np

from utilslibs.draw_tools import  draw_grid_graph_with_budget
''
if __name__ == "__main__":

    # this is for plot joint-tune-U

    y_k_array = [5, 10, 20, 50]
    x_epoch_array = [1, 5, 10, 50, 100]
    two_D_run_acc = [[0.9347, 0.9363, 0.9382, 0.9379], [0.9363, 0.9365, 0.9379, -1.0], [0.9365, 0.9363, -1.0, -1.0],
                     [0.9421, -1.0, -1.0, -1.0], [0.9421, -1.0, -1.0, -1.0]]
    two_D_run_BT = [[100.69678666666667, 110.42315333333333, 120.82666666666667, 185.08463333333336],
                    [105.79311333333335, 134.35286666666667, 167.05605333333335, -1.0],
                    [120.12115333333335, 200.84762666666666, -1.0, -1.0], [150.70705333333333, -1.0, -1.0, -1.0],
                    [236.06367333333333, -1.0, -1.0, -1.0]]
    two_D_run_b1 = [[15624.0, 15624.0, 15624.0, 15624.0], [15624.0, 15624.0, 15624.0, -1.0],
                    [15624.0, 15624.0, -1.0, -1.0], [15624.0, -1.0, -1.0, -1.0], [15624.0, -1.0, -1.0, -1.0]]
    two_D_run_b2 = [[5.0, 25.0, 50.0, 250.0], [19.0, 98.0, 199.0, -1.0], [58.0, 296.0, -1.0, -1.0],
                    [148.0, -1.0, -1.0, -1.0], [397.0, -1.0, -1.0, -1.0]]
    time_min = 300

    two_D_run_acc = np.array(two_D_run_acc) * 100
    draw_grid_graph_with_budget(
        two_D_run_acc.tolist(), two_D_run_BT, two_D_run_b1, two_D_run_b2, f"{time_min}b1b2",
        y_k_array, x_epoch_array
    )

    # draw_grid_graph_with_budget_only_Acc(
    #     two_D_run_acc.tolist(), two_D_run_BT, two_D_run_b1, two_D_run_b2,
    #     f"{time_min}b1b2_acc",
    #     y_k_array, x_epoch_array
    # )
