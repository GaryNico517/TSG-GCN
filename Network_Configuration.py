import Network_Trainer
import module.common_module as cm
from dataloader import Tooth_dataloader_Project_GCN
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--data_seed_list', required=False, default= '1,2,3,4,5')
parser.add_argument("--job_list", type=str,required=False, default= 'TSG-GCN')
parser.add_argument("--data_split_list", type=str,required=False, default= 'Adult_Tooth_quadrant')
parser.add_argument("--GNN_model", type=str,required=False, default= 'GCN')
parser.add_argument("--SEG_AM_DM", required=False, default= '1,1,0')
parser.add_argument("--Test_only", type=lambda x: x.lower() == 'true', default= False)
parser.add_argument("--num_epoch", type=int, required=False, default= 300)
parser.add_argument('--checkpoint_dir', type=str, required=False, default='model/TSG-GCN')
parser.add_argument('--model_load_path', type=str, required=False,default='model/TSG-GCN/save/best_model-1-0.1109-0.2452-0.0722-0.0000-0.0000.pth')
parser.add_argument('--model_load', type=bool, required=False,default=False)
# parser.add_argument("--folder_name", type=str, required=False, default= 'results_Tooth/')

args = parser.parse_args()

TSNE = False
str_ids = list(args.data_seed_list)

data_seed_list = []
for str_id in str_ids:
    if str_id == ',':
        continue
    id = int(str_id)
    if id >= 0:
        data_seed_list.append(id)

if ',' in args.job_list:
    job_list = args.job_list.split(',')
else:
    job_list = []
    job_list.append(args.job_list)

if ',' in args.data_split_list:
    data_split_list = args.data_split_list.split(',')
else:
    data_split_list = []
    data_split_list.append(args.data_split_list)

GNN_model = args.GNN_model

if ',' in args.SEG_AM_DM:
    SEG_AM_DM = args.SEG_AM_DM.split(',')
else:
    SEG_AM_DM = []
    SEG_AM_DM.append(args.SEG_AM_DM)

Test_only = bool(args.Test_only)
num_epoch = args.num_epoch

checkpoint_dir = args.checkpoint_dir
model_load_path = args.model_load_path
model_load = args.model_load


Val_dice = 0
val_dice = 0
test_results = 0

folder_name = 'result_Project'

for split in data_split_list:
    # Run jobs:
    for job in job_list:
        #for split in data_split_list[:]:
            Test_results = [0, 0, 0, 0]
            for seed in data_seed_list:
                print(split)
                if str(split) == 'Adult_Tooth_quadrant':
                    num_classes = 10
                    shape = [96, 160, 160]  # [96,160,160] [128,128,128]
                    device, data_sizes, modelDataloader = Tooth_dataloader_Project_GCN.AdultToothdata(seed, split,shape,num_classes)

                elif str(split) == 'Child_Tooth_quadrant':
                    num_classes = 16
                    shape = [128, 160, 160]  # [96,160,160] [128,128,128]
                    device, data_sizes, modelDataloader = Tooth_dataloader_Project_GCN.ChildToothdata(seed, split,shape,num_classes)

                else:
                    num_classes = 9
                    shape = [96, 160, 160]  # [96,160,160] [128,128,128]
                    device, data_sizes, modelDataloader = Tooth_dataloader_Project_GCN.AdultToothdata(seed, split,shape,num_classes)
                val_dice, test_results = Network_Trainer.network_training_epoch(Test_only, job, seed, split, device,
                                                                                               data_sizes, modelDataloader,num_classes,
                                                                                               num_epoch, shape,folder_name, TSNE, GNN_model,SEG_AM_DM, checkpoint_dir, model_load_path,model_load)
                # val_dice, test_results = Network_training_GCN_am.network_training_epoch(Test_only, job, seed, split, device,
                #                                                                                data_sizes, modelDataloader,num_classes,
                #                                                                                num_epoch, shape,folder_name, TSNE, AM_decoder,AM_only)

print('All PC Tooth GCN jobs finished')
