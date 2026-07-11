#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_50Hz_High" --modelPath "/home/users/j/jiawen/EVT/save/classifer_50Hz_High.pth" --condition "50hz_High" --batch_size 256
#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_50Hz_Low" --modelPath "/home/users/j/jiawen/EVT/save/classifer_50Hz_Low.pth" --condition "50hz_Low" --batch_size 256
#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_45Hz_High" --modelPath "/home/users/j/jiawen/EVT/save/classifer_45Hz_High.pth" --condition "45hz_High" --batch_size 256
#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_45Hz_Low" --modelPath "/home/users/j/jiawen/EVT/save/classifer_45Hz_Low.pth" --condition "45hz_Low" --batch_size 256
#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_40Hz_High" --modelPath "/home/users/j/jiawen/EVT/save/classifer_40Hz_High.pth" --condition "40hz_High" --batch_size 256
#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_40Hz_Low" --modelPath "/home/users/j/jiawen/EVT/save/classifer_40Hz_Low.pth" --condition "40hz_Low" --batch_size 256
#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_35Hz_High" --modelPath "/home/users/j/jiawen/EVT/save/classifer_35Hz_High.pth" --condition "35hz_High" --batch_size 256
#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_35Hz_Low" --modelPath "/home/users/j/jiawen/EVT/save/classifer_35Hz_Low.pth" --condition "35hz_Low" --batch_size 256
#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_30Hz_High" --modelPath "/home/users/j/jiawen/EVT/save/classifer_30Hz_High.pth" --condition "30hz_High" --batch_size 256
#python main.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_30Hz_Low" --modelPath "/home/users/j/jiawen/EVT/save/classifer_30Hz_Low.pth" --condition "30hz_Low" --batch_size 256

#python main.py --data_folder "/home/users/j/jiawen/datasets/paderborn/class13_0_12_1" --model_path "/home/users/j/jiawen/EVT/save/class13_0_12_1.pth" --epochs 200
#python main.py --data_folder "/home/users/j/jiawen/datasets/paderborn/class13_0_12_2" --model_path "/home/users/j/jiawen/EVT/save/class13_0_12_2.pth" --epochs 200
#python main.py --data_folder "/home/users/j/jiawen/datasets/paderborn/class13_0_12_3" --model_path "/home/users/j/jiawen/EVT/save/class13_0_12_3.pth" --epochs 200
#python main.py --data_folder "/home/users/j/jiawen/datasets/paderborn/class13_0_12_4" --model_path "/home/users/j/jiawen/EVT/save/class13_0_12_4.pth" --epochs 200


python Test.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_50Hz_High_test" --model_path "/home/users/j/jiawen/EVT/save/classifer_50Hz_High.pth" --save_path "/home/users/j/jiawen/EVT/FeatureMaps/class0_14_50Hz_High_pred" --condition "50hz_High"
python Test.py --data_folder "/home/users/j/jiawen/EVT/phm/class0_14_50Hz_Low_test" --model_path "/home/users/j/jiawen/EVT/save/classifer_50Hz_Low.pth" --save_path "/home/users/j/jiawen/EVT/FeatureMaps/class0_14_50Hz_Low_pred" --condition "50hz_Low"