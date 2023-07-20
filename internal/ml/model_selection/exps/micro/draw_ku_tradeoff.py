
from exps.draw_tab_lib import plot_heatmap

if __name__ == "__main__":

    # this is for plot joint-tune-U

    y_k_array = [5, 10, 20, 50]
    x_epoch_array = [1, 5, 10, 50, 100]
    two_D_run_acc = [[0.9347, 0.9363, 0.9382, 0.9379],
                     [0.9363, 0.9365, 0.9379, -1.0],
                     [0.9365, 0.9363, -1.0, -1.0],
                     [0.9421, -1.0, -1.0, -1.0],
                     [0.9421, -1.0, -1.0, -1.0]]

    two_D_run_BT = [[100.69678666666667, 110.42315333333333, 120.82666666666667, 185.08463333333336],
                    [105.79311333333335, 134.35286666666667, 167.05605333333335, -1.0],
                    [120.12115333333335, 200.84762666666666, -1.0, -1.0], [150.70705333333333, -1.0, -1.0, -1.0],
                    [236.06367333333333, -1.0, -1.0, -1.0]]

    two_D_run_b1 = [[15624.0, 15624.0, 15624.0, 15624.0], [15624.0, 15624.0, 15624.0, -1.0],
                    [15624.0, 15624.0, -1.0, -1.0], [15624.0, -1.0, -1.0, -1.0], [15624.0, -1.0, -1.0, -1.0]]
    two_D_run_b2 = [[5.0, 25.0, 50.0, 250.0], [19.0, 98.0, 199.0, -1.0], [58.0, 296.0, -1.0, -1.0],
                    [148.0, -1.0, -1.0, -1.0], [397.0, -1.0, -1.0, -1.0]]
    time_min = 300

    # Scale each value by 100 times
    scaled_data = [[value * 100 if value > 0 else value for value in row] for row in two_D_run_acc]
    plot_heatmap(data=scaled_data,
                 fontsize=18,
                 x_array_name="U ( # Training Epoch)",
                 y_array_name="K ( # Explored Models)",
                 title="Accuracy Achieved",
                 output_file="./internal/ml/model_selection/exp_result/300b1b2_ACC.pdf",
                 decimal_places=2)

    plot_heatmap(data=two_D_run_BT,
                 fontsize=18,
                 x_array_name="U ( # Training Epoch)",
                 y_array_name="K ( # Explored Models)",
                 title="Time Usage",
                 output_file="./internal/ml/model_selection/exp_result/300b1b2_T.pdf",
                 decimal_places=1
                 )

