




import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300

name_list = ['RSA','DES','3DES','AES']
num_list = [21,17,14,12]

# 配置填充的图形
patterns = ('/','//','-','\\')
fig = plt.figure(figsize=(8,6), dpi=72, facecolor="white")
axes = plt.subplot(111)
for i,pattern in enumerate(patterns):
    axes.bar( name_list[i], num_list[i], hatch=pattern, color='white',edgecolor='black',)
# 设置X轴上的文字
axes.set_xticks(name_list)
plt.savefig('abc.png')
plt.show()