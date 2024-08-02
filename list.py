import os

filepath = 'Typological Forms/fps_predict/type name.txt'

with open(filepath, 'r', encoding='utf-8') as file:
    type_list = [line.strip() for line in file]

filename = []
num_file = []

for type_name in type_list:
    directory = f'Typological Forms/fps_predict/{type_name}'
    # 获取目录下的所有文件名，并添加到 filename 列表中
    filenames = os.listdir(directory)
    filename.extend(filenames)
    num_file.append(str(len(filenames)))

# 将文件名写入txt文件
with open('fpstestfilenames.txt', 'w') as f:
    for name in filename:
        f.write(name + '\n')

# with open('filenums.txt', 'w') as f:
#     for num in num_file:
#         f.write(num + '\n')

