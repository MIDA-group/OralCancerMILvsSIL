import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import argparse
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os, fnmatch
import cv2
import shutil
import sklearn
from sklearn import metrics
from pathlib import Path
from dataloaderOC import OC_SIL
from models import ModelParallelResNet18, Lenet, ModelParallelSqueezeNet
from aux_functions import list_files_in_folder, create_save_dir, create_list_of_files_in_given_folder, working_dataset,list_files_oral_cancer3_focused, transformations_dataset_oral_cancer3_focused, load_dataset, postproc, _preprocess, postproc_new_separate_distrib_pos_neg_bags_train_valid_for_test, find



# # Training settings
parser = argparse.ArgumentParser(description='PyTorch PAPQMNIST/OC bags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0005)') #0.0001
parser.add_argument('--reg', type=float, default=10e-6, metavar='R',
                    help='weight decay') 
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model_name', type=str, default='1.pt', metavar='M',
                    help='random seed (default: 1)')
parser.add_argument('--tmpdir', type=str, default='./', help='$TMPDIR')
# parser.add_argument('--output_dir', type=str, default='./', help='Save outputs')
parser.add_argument('--input_dir', type=str, default='./', help='Dataset folder')
parser.add_argument('--avg_thres', type=float, default=-100, metavar='Th',
                    help='average valid threshold 9 folds') 
args, unknown = parser.parse_known_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')
batch_size = 56 
num_classes = 2
n_channels = 3
num_workers = 0
img_width, img_height = 80, 80 
# datasetOC = False

list_posit_bags = ['01', '05', '53', '07', '37','55', '86','88','98','03','101','96']
list_negat_bags = ['26', '61', '78', '59', '63','80', '68','71','75','65','73','70']

name_class = ['negative', 'positive']
folds = [1]
tmpdir0='../../fold'
valid_thresholds = {}
valid_thresholds[1] = 0.02 #example
# percent_key_ins = 10
# dataset_name = 'PAPQMNIST_0012__beta_distr_1'
# dataset_name = f'PAPQMNIST_0012__{percent_key_ins:04d}_1'
args.input_dir = 'OC_dataset_mock' #'PAPQMNIST_0012__0005_1' 


# test_path_bags = os.path.join(root, dataset_name, 'test')
num_test_bags = 8
gpu_number = '0'

# mean, std of dataset
mean_dataset = [0.5223, 0.6717, 0.7039] # example
std_dataset = [0.1364, 0.1376, 0.1370] # example
model_architecture = 'squeezenet' # resnet18, squeezenet, lenet

TRAIN = 'train'
VAL = 'valid'
TEST = 'test'
if args.cuda:
    if model_architecture=='resnet18':
        model = ModelParallelResNet18(gpu_number) 
    elif model_architecture=='squeezenet':
        model = ModelParallelSqueezeNet(gpu_number)
    elif model_architecture=='lenet':
        model = Lenet(gpu_number) 
window_length = 15 # valudation window of epochs for calculating moving average    
# Parameters for loader
params = {'dim': (img_width, img_height),
          'batch_size': batch_size,
          'n_classes': 2,
          'negative_class_name': 'negative',
          'positive_class_name': 'positive',
          'n_channels': n_channels,
          'shuffle': True}
    
for i in folds: 
    args.tmpdir = tmpdir0+str(i)
    args.avg_thres = valid_thresholds[i]
    test_path_bags = os.path.join(args.tmpdir, args.input_dir, 'test')
    print(test_path_bags, args.tmpdir, args.input_dir)
    # Enter inputs
    data_dir = os.path.join(args.tmpdir,args.input_dir+'_SIL') #os.path.join(root, basePath.parts[-1]+'_SIL')
    
    ### From MIL to SIL dataset
    data_folder = data_dir[:-4] #os.path.join(root, dataset_name)
    basePath = Path(data_folder)
    save_dir = create_save_dir(basePath.parent.absolute(), basePath.parts[-1]+'_SIL')
    if len(os.listdir(save_dir)) == 0:
        for child in basePath.iterdir():
            if child.is_dir() and (child.parts[-1]=='train' or child.parts[-1]=='valid' or child.parts[-1]=='test'):
                set_folder = create_save_dir(save_dir, child.parts[-1])
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        class_folder = create_save_dir(set_folder, grandchild.parts[-1])
                        for ggrandchild in grandchild.iterdir():
                            if ggrandchild.is_dir():
                                files_names = list_files_in_folder(ggrandchild)
                                for i in range(len(files_names)):
                                    src = os.path.join(ggrandchild, files_names[i])
                                    dst = os.path.join(class_folder, files_names[i])
                                    shutil.copy(src, dst)





    data_transforms = transformations_dataset_oral_cancer3_focused(TRAIN, VAL, TEST, img_width, img_height, mean_dataset,
                                                                   std_dataset)
    dataset_sizes = {x: len(working_dataset(x, data_dir, data_transforms)) for x in [TRAIN, VAL, TEST]}
    for x in [TRAIN, VAL]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))
    print("Classes: ")
    class_names = working_dataset(TRAIN, data_dir, data_transforms).classes
    print(class_names)
    training_generator = load_dataset(working_dataset(TRAIN, data_dir, data_transforms), batch_size,
                                      num_workers)
    train_iter = iter(training_generator)
    valid_generator = load_dataset(working_dataset(VAL, data_dir, data_transforms), batch_size, num_workers)
    test_generator = load_dataset(working_dataset(TEST, data_dir, data_transforms), batch_size, num_workers)

    files_train, files_test, files_valid, Labels = list_files_oral_cancer3_focused(data_dir, TRAIN, VAL, TEST,
                                                                      negative_class='negative',
                                                                      positive_class='positive')

    partition = {'train': files_train, 'test': files_test, 'valid': files_valid}  # IDs
    training_generator = OC_SIL(partition['train'], Labels, data_transforms[TRAIN],
                                                    path=os.path.join(data_dir, TRAIN), **params)
    valid_generator = OC_SIL(partition['valid'], Labels, data_transforms[VAL],
                                                path=os.path.join(data_dir, VAL), **params)
    test_generator = OC_SIL(partition['test'], Labels, data_transforms[TEST],
                                                path=os.path.join(data_dir, TEST), **params)
    print('Init Model')


    save_weights_dir = create_save_dir(os.path.join(data_dir), 'test_'+
                                       '_train_epochs_'+str(args.epochs)+'_lr'+str(args.lr)+'_model_'+model_architecture+'SIL') 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)




    path = Path(data_dir)
    all_train_loss = []; all_train_acc=[]; all_valid_loss=[]; all_train_error=[];all_valid_acc=[]; all_train_F1score=[];all_valid_F1score=[]
    all_valid_error=[]
    count_stable_epochs=[]; 
    min_valid_loss = torch.tensor([float('inf')]).to('cuda:'+gpu_number) 
    idx_model_in_window_with_min_loss=0
    name_best_inMovAv_val_error_model='1'
    min_valid_error = np.inf; window_with_best_avg=np.inf
    num_epochs=args.epochs; lr=args.lr

    train_batches = len(training_generator)
    val_batches = len(valid_generator)
    for epoch in range(1, args.epochs + 1):
        print("Training started")
        model.train()
        train_loss = 0.; train_error = 0.
        TN = 0; TP = 0; FN = 0; FP = 0; TN_train = 0; TP_train = 0; FN_train = 0; FP_train = 0
        loss_train = 0;loss_valid = 0
        acc_train = 0;acc_valid = 0
        for i, data_all in enumerate(training_generator):
            if i >= train_batches:
                break
            data, label = data_all
            if args.cuda:
                data, label = data.to('cuda:'+gpu_number), label.to('cuda:'+gpu_number) 
            data, label = Variable(data), Variable(label)

            label = label.long()
            optimizer.zero_grad()
            outputs = model(data)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            acc_train += (torch.sum(preds == label.data)).item()
            for bt_nr in range(len(preds)):
                if label.data[bt_nr]==0:
                    if preds[bt_nr]==0:
                        TN_train += 1
                    elif preds[bt_nr]==1:
                        FP_train += 1
                    else:
                        print(preds[bt_nr])
                elif label.data[bt_nr]==1:
                    if preds[bt_nr]==1:
                        TP_train += 1
                    elif preds[bt_nr]==0:
                        FN_train += 1
                    else:
                        print(preds[bt_nr])
                else:
                    print("LABELS",label.data[bt_nr])
        # F1score:
        Pr_train = TP_train/(TP_train+FP_train+1e-12)
        Re_train = TP_train/(TP_train+FN_train+1e-12)
        F1_score_train = 2*(Pr_train*Re_train)/(Pr_train+Re_train+1e-12)
        train_loss = loss_train / len(training_generator.dataset)
        train_acc = acc_train / len(training_generator.dataset)    
        # Validation loss and error
        valid_loss = 0.; valid_error = 0.
        model.eval()
        for i, data_all in enumerate(valid_generator):
            if i >= val_batches:
                break
            data, label = data_all
            if args.cuda:
                data, label = data.to('cuda:'+gpu_number), label.to('cuda:'+gpu_number) 
            data, label = Variable(data), Variable(label) 
            label = label.long()
            optimizer.zero_grad()
            outputs = model(data)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, label)
            loss_valid += loss.item()
            acc_valid += (torch.sum(preds == label.data)).item()
            for bt_nr in range(len(preds)):
                if label.data[bt_nr]==0:
                    if preds[bt_nr]==0:
                        TN += 1
                    elif preds[bt_nr]==1:
                        FP += 1
                    else:
                        print(preds[bt_nr])
                elif label.data[bt_nr]==1:
                    if preds[bt_nr]==1:
                        TP += 1
                    elif preds[bt_nr]==0:
                        FN += 1
                    else:
                        print(preds[bt_nr])
                else:
                    print("LABELS",label.data[bt_nr])
            Conf_matrix = np.stack((TN,FP,FN,TP))       
        # F1score:
        Pr = TP/(TP+FP+1e-12)
        Re = TP/(TP+FN+1e-12)
        F1_score = 2*(Pr*Re)/(Pr+Re+1e-12)
        valid_loss = loss_valid / len(valid_generator.dataset)
        valid_acc = acc_valid / len(valid_generator.dataset) 

        # Save all losses for all epochs both valid and train
        all_train_loss.append(train_loss) #(train_loss.cpu().data.numpy())
        all_train_F1score.append(F1_score_train)
        all_valid_loss.append(valid_loss)
        all_valid_F1score.append(F1_score)
        all_train_acc.append(train_acc)
        all_valid_acc.append(valid_acc)
        all_train_error.append(1/(F1_score_train+10e-12))
        all_valid_error.append(1/(F1_score+10e-12))

        # Best validation F1score, within the best average window of F1score
        name_epoch_model = os.path.join(path, 'saved_model_OC_'+path.parts[-1]+'_currentEpoch'+\
                                        str(epoch)+'_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.pt')
        torch.save(model.state_dict(), name_epoch_model)
        if len(all_valid_error)>window_length-1:
            avg_valid_error_window = np.mean(np.asarray(all_valid_error)[-window_length:])

            if window_with_best_avg > avg_valid_error_window and (epoch-window_length+idx_model_in_window_with_min_loss+1)>20:
                idx_model_in_window_with_min_loss = np.argmin(np.asarray(all_valid_error)[-window_length:])
                window_with_best_avg = avg_valid_error_window
                name_best_inMovAv_val_error_model = os.path.join(path, 'saved_model_OC_'+path.parts[-1]+
                                                                '_currentEpoch'+
                                                 str(epoch-window_length+idx_model_in_window_with_min_loss+1)+
                                                 '_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+
                                                 '.pt')
                print("Current best moving average model, epoch: ", str(epoch-window_length+idx_model_in_window_with_min_loss+1))
                epoch_best_movingAvg = epoch-window_length+idx_model_in_window_with_min_loss+1
        #to delete models that are outside current window except the best_window_model_name
        for ep in range(0, epoch-window_length):
            model_nametmp = os.path.join(path, 'saved_model_OC_'+path.parts[-1]+'_currentEpoch'+
                                         str(ep+1)+'_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+
                                         '.pt')
            if model_nametmp != name_best_inMovAv_val_error_model and os.path.isfile(model_nametmp):
                os.remove(model_nametmp)

        print('Epoch: {}, Loss: {:.4f}, Train F1: {:.4f}'.format(epoch, train_loss, F1_score_train))
        print('Epoch: {}, Loss: {:.4f}, Valid F1: {:.4f}'.format(epoch, valid_loss, F1_score))


    new_name_best_inMovAv_val_error_model = os.path.join(save_weights_dir, 'saved_model_best_movingAvg_OC_'+\
                                                         path.parts[-1]+'epochSaved_'+str(epoch_best_movingAvg)+
                                                         '_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+
                                                         '.pt')
    shutil.move(name_best_inMovAv_val_error_model, new_name_best_inMovAv_val_error_model)


    for ep in range(1,num_epochs+1):
        model_nametmp = os.path.join(path, 'saved_model_OC_'+path.parts[-1]+'_currentEpoch'+str(ep)+
                                     '_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.pt')
        if os.path.isfile(model_nametmp):
            os.remove(model_nametmp)

    np.save(os.path.join(save_weights_dir,'all_train_loss_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.npy'), np.asarray(all_train_loss))
    np.save(os.path.join(save_weights_dir,'all_valid_loss_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.npy'), np.asarray(all_valid_loss))
    np.save(os.path.join(save_weights_dir,'all_train_error_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.npy'), np.asarray(all_train_error))
    np.save(os.path.join(save_weights_dir,'all_valid_error_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.npy'), np.asarray(all_valid_error))              
    
    # Inference
    model_name = find('*.pt', save_weights_dir)
    if len(model_name)==1:
        args.model_name = model_name[0]
    else:
        "Error: Two models saved?" 
    model.load_state_dict(torch.load(args.model_name))
    model_path = Path(args.model_name)
    model.eval()
    device = 'cuda:'+gpu_number
    
    if args.avg_thres==-100: # valid or train set, find threshold for a fold
        pred_bag_labels_negat_valid, pred_bag_labels_posit_valid, acc_bag = postproc_new_separate_distrib_pos_neg_bags_train_valid_for_test('valid', data_dir, data_transforms, name_class, save_weights_dir, model_path, device, VAL, model, test_path_bags, list_posit_bags, list_negat_bags, args.avg_thres)
        mean_perc_as_pos_in_neg_val_bags = np.mean(np.asarray(pred_bag_labels_negat_valid))
        mean_perc_as_pos_in_pos_val_bags = np.mean(np.asarray(pred_bag_labels_posit_valid))
        diffefence = mean_perc_as_pos_in_pos_val_bags-mean_perc_as_pos_in_neg_val_bags
        threshold_mean_valid = diffefence/2+mean_perc_as_pos_in_neg_val_bags   
        print('Threshold of the fold based on mean, valid: ', threshold_mean_valid)
    else: # test set
        pred_bag_labels_negat, pred_bag_labels_posit, acc_bag = postproc_new_separate_distrib_pos_neg_bags_train_valid_for_test('test', data_dir, data_transforms, name_class, save_weights_dir, model_path, device, TEST, model, test_path_bags, list_posit_bags, list_negat_bags, args.avg_thres) #

        # save N top score for each patient with malignancy
        cl_ind = 1
        top_number=25
        output_dir = os.path.join(save_weights_dir, 'test', name_class[cl_ind], name_class[cl_ind] + '_'+ model_path.parts[-1])
        top36dir = create_save_dir(os.path.join(save_weights_dir, 'test', name_class[cl_ind],name_class[cl_ind] + '_'+ model_path.parts[-1]), 'top'+str(top_number))
        score_names = list_files_in_folder(os.path.join(output_dir))

        list_bags = list_posit_bags
        for b in range(len(list_bags)):
            bag_folder = os.path.join(test_path_bags, name_class[cl_ind], list_bags[b])
            if os.path.exists(os.path.join(bag_folder)):
                save_36 = create_save_dir(top36dir, list_bags[b])
                bag_folder_names = list_files_in_folder(bag_folder)  
            
                bag_pos_scores=[]; bag_names=[]
                for i in range(len(score_names)):
                    if score_names[i][:-4] in bag_folder_names:
                        score_temp = np.load(os.path.join(output_dir, score_names[i]))
                        bag_pos_scores.append(score_temp)
                        bag_names.append(score_names[i])
                bag_pos_scores_array = np.asarray(bag_pos_scores)
                idxs = np.flip(np.argsort(bag_pos_scores_array)) 
                top36 = []
                for i in range(top_number):
                    top36.append(bag_names[idxs[i]])
                    src=os.path.join(data_dir, 'test', name_class[cl_ind], bag_names[idxs[i]][:-4])
                    dst=os.path.join(save_36, bag_names[idxs[i]][:-4])
                    shutil.copy(src, dst)
                np.save(os.path.join(save_36,'top'+ str(top_number) +'.npy'),np.asarray(top36))
                

        print('num_test_bags:',num_test_bags)
        print('Accuracy bag level:', acc_bag)       
        
        
        
