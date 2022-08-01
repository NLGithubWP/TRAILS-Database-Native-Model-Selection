

# #
# contents = []
# for i in range(1,9,1):
#     fileName = "file-" + str(i)
#     f = open(fileName,'r')
#     content = f.readlines()
#     contents.append(content)
#
# mapper = {}
# total = 0
#
# for content in contents:
#     for line in content:
#         if "201union_best total_pair = " in line:
#             total += int(line.split(" = ")[1])
#         if "]," in line:
#             res = line.split("],")
#             num = int(res[1].split(" / ")[0].strip())
#
#             if res[0] in mapper:
#                 mapper[res[0]] += num
#             else:
#                 mapper[res[0]] = num
#
#
# mapper2 = {k: v for k, v in sorted(mapper.items(), key=lambda item: item[1])}
#
# for ele, value in mapper2.items():
#     print(len(ele.split(",")), ele,  "value/total = ", value, "/",total, "=" ,value/total)
#










