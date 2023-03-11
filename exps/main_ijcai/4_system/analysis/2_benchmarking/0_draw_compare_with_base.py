from common.constant import Config
from utilslibs.io_tools import read_json
from utilslibs.draw_tools import get_plot_compare_with_base_line_cfg, get_base_annotations


dataset = Config.c10
search_space = Config.NB201

if dataset == Config.imgNet:
    img_in_graph = "ImageNet"
else:
    img_in_graph = dataset

saved_dict = read_json(f"./0_macro_res_{search_space}_{dataset}")

run_range_, budget_array, sub_graph_y1, sub_graph_y2, sub_graph_split, draw_graph = \
    get_plot_compare_with_base_line_cfg(search_space, dataset, True)

# this is firmest
y_acc_list_arr = saved_dict["y_acc_list_arr"]
x_T_list = saved_dict["x_T_list"]
# this is training-based
x_acc_train = saved_dict["x_acc_train"]
y_acc_train_l = saved_dict["y_acc_train_l"]
y_acc_train_m = saved_dict["y_acc_train_m"]
y_acc_train_h = saved_dict["y_acc_train_h"]
# this is firmest-p1
y_acc_list_arr_only_phase1 = saved_dict["y_acc_list_arr_only_phase1"]
x_T_list_only_phase1 = saved_dict["x_T_list_only_phase1"]


def sample_some_points(
        time_array, value_array, value_p1_array,
        train_x_time_array,
        train_x_acc_array_low, train_x_acc_array_mean, train_x_acc_array_high):

    new_time_array = []
    new_value_array = []
    new_value_p1_array = []

    new_train_x_time_array = []
    new_train_x_acc_array_low = []
    new_train_x_acc_array_mean = []
    new_train_x_acc_array_high = []

    if dataset == Config.c10:
        new_time_array = [0.017, 1, 5, 9, 17, 33, 65, 101, 201, 301, 349]
        new_index_array = [time_array.index(ele) for ele in new_time_array]
        for ori_value in value_array:
            new_value_array.append([ori_value[i] for i in new_index_array])
        for ori_value in value_p1_array:
            new_value_p1_array.append([ori_value[i] for i in new_index_array])
        new_train_x_time_array = [1.937728786515811, 5.659422259056379, 11.30026762554097,
                                  32.55173002024965, 64.55501621393931, 109.55948316201805,
                                  199.41135477142203, 259.6101172423315
                                  ]
        new_index_array_train_time = [train_x_time_array.index(ele) for ele in new_train_x_time_array]
        new_train_x_acc_array_low = [train_x_acc_array_low[i] for i in new_index_array_train_time]
        new_train_x_acc_array_mean = [train_x_acc_array_mean[i] for i in new_index_array_train_time]
        new_train_x_acc_array_high = [train_x_acc_array_high[i] for i in new_index_array_train_time]

    elif dataset == Config.c100:
        new_time_array = [0.017, 1, 5, 9,  21, 33, 65, 101, 201, 301, 401, 501, 601]
        new_index_array = [time_array.index(ele) for ele in new_time_array]
        for ori_value in value_array:
            new_value_array.append([ori_value[i] for i in new_index_array])
        for ori_value in value_p1_array:
            new_value_p1_array.append([ori_value[i] for i in new_index_array])
        new_train_x_time_array = [3.5614719922107363, 7.230048829222483, 10.87099102103994, 32.70721482154396,
                                  63.587714490578286, 99.69352991952312, 199.54702036260142, 335.11532325574325,
                                  500.7024714156276, 571.742354752478]
        new_index_array_train_time = [train_x_time_array.index(ele) for ele in new_train_x_time_array]

        new_train_x_acc_array_low = [train_x_acc_array_low[i] for i in new_index_array_train_time]
        new_train_x_acc_array_mean = [train_x_acc_array_mean[i] for i in new_index_array_train_time]
        new_train_x_acc_array_high = [train_x_acc_array_high[i] for i in new_index_array_train_time]

    else:
        new_time_array = [0.017, 1, 9, 49, 105, 201, 401, 601, 801, 1001, 1401, 1801, 1993]
        new_index_array = [time_array.index(ele) for ele in new_time_array]
        for ori_value in value_array:
            new_value_array.append([ori_value[i] for i in new_index_array])
        for ori_value in value_p1_array:
            new_value_p1_array.append([ori_value[i] for i in new_index_array])
        new_train_x_time_array = [10.727976801193897, 32.94746213361422, 65.92062850949083,
                                  100.81394913731927, 203.56770449207792, 399.9518908892631, 606.4884816085666, 804.2524871640295, 1005.9584948723733,
                                  1339.5562870200365, 1765.9798337116295
                                  ]
        new_index_array_train_time = [train_x_time_array.index(ele) for ele in new_train_x_time_array]

        new_train_x_acc_array_low = [train_x_acc_array_low[i] for i in new_index_array_train_time]
        new_train_x_acc_array_mean = [train_x_acc_array_mean[i] for i in new_index_array_train_time]
        new_train_x_acc_array_high = [train_x_acc_array_high[i] for i in new_index_array_train_time]

    return new_time_array, new_value_array, new_value_p1_array, new_train_x_time_array,\
           new_train_x_acc_array_low, new_train_x_acc_array_mean, new_train_x_acc_array_high


x_T_list, y_acc_list_arr, y_acc_list_arr_only_phase1, x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h = \
    sample_some_points(
        x_T_list, y_acc_list_arr, y_acc_list_arr_only_phase1, x_acc_train,
        y_acc_train_l, y_acc_train_m, y_acc_train_h)

draw_graph(y_acc_list_arr, x_T_list, y_acc_list_arr_only_phase1, x_T_list,
           x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h,
           get_base_annotations(dataset), sub_graph_split,
           f"{search_space}_{dataset}", img_in_graph,
           sub_graph_y1,
           sub_graph_y2)
