

import json

from check_sampler import draw_sampler_res_sub


input_file = './101/compare_sampler_file'
input_file2 = './bohb'


with open(input_file, 'r') as readfile:
    dataBest = json.load(readfile)


draw_sampler_res_sub(dataBest, "./101/compare_sampler_file.jpg")
