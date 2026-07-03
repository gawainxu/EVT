import argparse
import os
import shutil

def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--condition', type=str, required=True, default="50Hz_High", help="condition condition")

    return parser.parse_args()


opt = arg_parse()

labelDict = ["helical 1_" + opt.condition + "_1", "helical 1_" + opt.condition + "_2",
             "helical 2_" + opt.condition + "_1", "helical 2_" + opt.condition + "_2",
             "helical 3_" + opt.condition + "_1", "helical 3_" + opt.condition + "_2",
             "helical 4_" + opt.condition + "_1", "helical 4_" + opt.condition + "_2",
             "helical 5_" + opt.condition + "_1", "helical 5_" + opt.condition + "_2",
             "helical 6_" + opt.condition + "_1", "helical 6_" + opt.condition + "_2",
             "spur 1_" + opt.condition + "_1", "spur 1_" + opt.condition + "_2",]


outLabels = ["spur 2_" + opt.condition + "_1", "spur 2_" + opt.condition + "_2",
             "spur 3_" + opt.condition + "_1", "spur 3_" + opt.condition + "_2",
             "spur 4_" + opt.condition + "_1", "spur 4_" + opt.condition + "_2",
             "spur 5_" + opt.condition + "_1", "spur 5_" + opt.condition + "_2",
             "spur 6_" + opt.condition + "_1", "spur 6_" + opt.condition + "_2",
             "spur 7_" + opt.condition + "_1", "spur 7_" + opt.condition + "_2",
             "spur 8_" + opt.condition + "_1", "spur 8_" + opt.condition + "_2"]


source_folder = "'/home/users/j/jiawen/EVT/phm/class0_28_" + opt.condition
dist_folder = "'/home/users/j/jiawen/EVT/phm/class_outliers_" + opt.condition
if not os.path.exists(dist_folder):
    os.mkdir(dist_folder)

file_list = sorted(os.listdir(source_folder))

for fn in file_list:
    print(fn)
    label_name = fn.split("__")[0]
    if label_name in labelDict:
        continue

    shutil.move(os.path.join(source_folder, fn), os.path.join(dist_folder, label_name))
