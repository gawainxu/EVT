#python imgTS.py --condition 50Hz_High
#python imgTS.py --condition 50Hz_Low
#python imgTS.py --condition 45Hz_High
#python imgTS.py --condition 45Hz_Low
#python imgTS.py --condition 40Hz_High
#python imgTS.py --condition 40Hz_Low
#python imgTS.py --condition 35Hz_High
#python imgTS.py --condition 35Hz_Low
#python imgTS.py --condition 30Hz_High
#python imgTS.py --condition 30Hz_Low

#python setSplit.py --condition_name 50hz_High --condition 50Hz_High
#python setSplit.py --condition_name 50hz_Low --condition 50Hz_Low
#python setSplit.py --condition_name 45hz_High --condition 45Hz_High
#python setSplit.py --condition_name 45hz_Low --condition 45Hz_Low
#python setSplit.py --condition_name 40hz_High --condition 40Hz_High
#python setSplit.py --condition_name 40hz_Low --condition 40Hz_Low
#python setSplit.py --condition_name 35hz_High --condition 35Hz_High
#python setSplit.py --condition_name 35hz_Low --condition 35Hz_Low
#python setSplit.py --condition_name 30hz_High --condition 30Hz_High
#python setSplit.py --condition_name 30hz_Low --condition 30Hz_Low


python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_30Hz_High" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_30Hz_High_test"
python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_30Hz_Low" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_30Hz_Low_test"
python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_35Hz_High" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_35Hz_High_test"
python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_35Hz_Low" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_35Hz_Low_test"
python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_40Hz_High" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_40Hz_High_test"
python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_40Hz_Low" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_40Hz_Low_test"
python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_45Hz_High" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_45Hz_High_test"
python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_45Hz_Low" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_45Hz_Low_test"
python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_50Hz_High" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_50Hz_High_test"
python train_test_split.py --train_dir "/home/users/j/jiawen/EVT/phm/class0_14_50Hz_Low" --test_dir "/home/users/j/jiawen/EVT/phm/class0_14_50Hz_Low_test"

