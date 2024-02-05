import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import transforms
import argparse
import torch.optim as optim
from torch.autograd import Variable
import os
import shutil
from pathlib import Path
from functions_PAPQMNIST import list_files_in_folder, create_save_dir, PAPQMNIST_Bags, AddGaussianNoise, count_max_num_instances, test, test_bags_info, find
from attention_models import Attention_bags_1GPU
from evaluationPAPQMNIST import compute_metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
    
""" Enter inputs """
parser = argparse.ArgumentParser(description='PyTorch PAPQMNIST bags Example')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.000005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay') #10e-5
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--sampling_size', type=int, default=100, metavar='SamplingSize',
                    help='sampling size in number of instances to sample')
args, unknown = parser.parse_known_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

# Positive bags list:
list_posit_bags = ['01', '05', '53', '07', '37','55', '86','88','98','03','101','96'] # list of positive bags
# percent_key_ins = 10 # percent of key instances in PAPQMNIST
data_path = '../../fold1/PAPQMNIST_0012__0005_1'  #f'../fold1/PAPQMNIST_0012__{percent_key_ins:04d}_1'
mean_dataset = [0.5223, 0.6717, 0.7039] # example
std_dataset = [0.1364, 0.1376, 0.1370] # example
model_architecture = 'resnet18' # resnet18, squeezenet, lenet
gpu_number = '0'

if model_architecture=='lenet':
    args.lr=0.00005; args.reg=10e-6
elif model_architecture=='resnet18':
    args.lr=0.000005; args.reg=10e-5
elif model_architecture=='squeezenet':
    args.lr=0.00005; args.reg=10e-5


path = Path(data_path)
    
# Largest number of instances per bag
max_bag_length = count_max_num_instances(data_path)
print(max_bag_length)
sampling_size_in_instances = args.sampling_size 
if max_bag_length>sampling_size_in_instances: 
    # use sampling
    test_epochs = np.ceil((10*max_bag_length)/sampling_size_in_instances).astype(int); val_times = 5
else:
    # no sampling
    test_epochs = 1; val_times = 1
print(test_epochs,'test_epochs')
window_length = 15 # valudation window of epochs for calculating moving average

save_weights_dir = create_save_dir(os.path.join(data_path), 'test_weights_epochs_'+str(test_epochs)+'samplingSize_'+str(args.sampling_size)+'_train_epochs_'+str(args.epochs)+
                                   '_lr'+str(args.lr)+'_model_'+model_architecture+'MIL') 


# '''
print('Init Model')
if args.cuda:
    model = Attention_bags_1GPU(gpu_number, model_architecture)
    model.to('cuda:'+gpu_number) 

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

train_loader = data_utils.DataLoader(PAPQMNIST_Bags(train=True,
                                               valid=False,
                                               test=False,
                                     transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                         transforms.RandomVerticalFlip(p=0.5),                                 
                                    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                                    transforms.ToTensor(),
                                    transforms.RandomApply(transforms=[AddGaussianNoise(0, var1=2.0, var2=5.0)], p=0.15), 
                                                        transforms.Normalize(mean=mean_dataset, 
                                                                                        std=std_dataset),
                                                                   
                                     ]), sampling_size=sampling_size_in_instances, list_positive_bags = list_posit_bags,
                                                    data_path=data_path), 
                                     batch_size=1,
                                     shuffle=True)

valid_loader = data_utils.DataLoader(PAPQMNIST_Bags(train=False,
                                              valid=True,
                                              test=False,
                                     transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                         transforms.RandomVerticalFlip(p=0.5),                                 
                                    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                                    transforms.ToTensor(),
                                    transforms.RandomApply(transforms=[AddGaussianNoise(0, var1=2.0, var2=5.0)], p=0.15), 
                                                        transforms.Normalize(mean=mean_dataset, 
                                                                                        std=std_dataset),
                                                                   
                                     ]), sampling_size=sampling_size_in_instances, list_positive_bags = list_posit_bags,
                                                    data_path=data_path),
                                    batch_size=1,
                                    shuffle=False)

''' 
""" Train """
''' 
path = Path(data_path)
all_train_loss = []; all_train_error=[]; all_valid_loss=[]; all_valid_error=[]
count_stable_epochs=[]; name_best_inMovAv_val_error_model='1'
min_valid_loss = torch.tensor([float('inf')]).to('cuda:'+gpu_number) 
min_valid_error = np.inf; window_with_best_avg=np.inf
num_epochs=args.epochs; lr=args.lr
        
for epoch in range(1, args.epochs + 1):
    print("Training started")
    model.train()
    train_loss = 0.; train_error = 0.
    for batch_idx, (data, label, sample_indices, index, bages_names) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.to('cuda:'+gpu_number), bag_label.to('cuda:'+gpu_number) 
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data[0]
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error

        # backward pass
        loss.backward()
        optimizer.step()
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    
    # Validation loss and error
    valid_loss = 0.; valid_error = 0.
    model.eval()
    for epp in range(0, val_times):
        for batch_idx, (data, label, sample_indices, index, bages_names) in enumerate(valid_loader):
            bag_label = label[0]
            if args.cuda:
                data, bag_label = data.to('cuda:'+gpu_number), bag_label.to('cuda:'+gpu_number) 
            data, bag_label = Variable(data), Variable(bag_label)  
            loss, _ = model.calculate_objective(data, bag_label)
            valid_loss += loss.data[0]
            error, _ = model.calculate_classification_error(data, bag_label)
            valid_error += error

    valid_loss /= len(valid_loader)*val_times
    valid_error /= len(valid_loader)*val_times
    
    # Save all losses for all epochs both valid and train
    all_train_loss.append(train_loss.cpu().data.numpy())
    all_train_error.append(train_error)
    all_valid_loss.append(valid_loss.cpu().data.numpy())
    all_valid_error.append(valid_error)
    
    # Best validation error within the best average window of error
    name_epoch_model = os.path.join(os.path.join(data_path), 'saved_model_PAPQMNIST_'+model_architecture+path.parts[-1]+'_currentEpoch'+\
                                    str(epoch)+'_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'_sampSize'+\
                                    str(sampling_size_in_instances)+'.pt')
    torch.save(model.state_dict(), name_epoch_model)
    if len(all_valid_error)>window_length-1:
        avg_valid_error_window = np.mean(np.asarray(all_valid_error)[-window_length:])
        
        if window_with_best_avg > avg_valid_error_window and epoch>20:
            idx_model_in_window_with_min_loss = np.argmin(np.asarray(all_valid_error)[-window_length:])
            window_with_best_avg = avg_valid_error_window
            name_best_inMovAv_val_error_model = os.path.join(os.path.join(data_path), 'saved_model_PAPQMNIST_'+model_architecture+path.parts[-1]+
                                                            '_currentEpoch'+
                                             str(epoch-window_length+idx_model_in_window_with_min_loss+1)+
                                             '_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+
                                             '_sampSize'+str(sampling_size_in_instances)+'.pt')
            print("Current best moving average model, epoch: ", str(epoch-window_length+idx_model_in_window_with_min_loss+1))
            epoch_best_movingAvg = epoch-window_length+idx_model_in_window_with_min_loss+1
    #to delete models that are outside current window except the best_window_model_name
    for ep in range(0, epoch-window_length):
        model_nametmp = os.path.join(os.path.join(data_path), 'saved_model_PAPQMNIST_'+model_architecture+path.parts[-1]+'_currentEpoch'+
                                     str(ep+1)+'_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'_sampSize'+
                                     str(sampling_size_in_instances)+'.pt')
        if model_nametmp != name_best_inMovAv_val_error_model and os.path.isfile(model_nametmp):
            os.remove(model_nametmp)
    
    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))
    print('Epoch: {}, Loss: {:.4f}, Valid error: {:.4f}'.format(epoch, valid_loss.cpu().numpy()[0], valid_error))

    
new_name_best_inMovAv_val_error_model = os.path.join(save_weights_dir, 'saved_model_best_movingAvg_PAPQMNIST_'+model_architecture+\
                                                     path.parts[-1]+'epochSaved_'+str(epoch_best_movingAvg)+
                                                     '_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+
                                                     '_sampSize'+str(sampling_size_in_instances)+'.pt')
shutil.move(name_best_inMovAv_val_error_model, new_name_best_inMovAv_val_error_model)
    
    
for ep in range(1,num_epochs+1):
    model_nametmp = os.path.join(os.path.join(data_path), 'saved_model_PAPQMNIST_'+model_architecture+path.parts[-1]+'_currentEpoch'+str(ep)+'_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'_sampSize'+str(sampling_size_in_instances)+'.pt')
    if os.path.isfile(model_nametmp):
        os.remove(model_nametmp)

np.save(os.path.join(save_weights_dir,'all_train_loss_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.npy'), np.asarray(all_train_loss))
np.save(os.path.join(save_weights_dir,'all_valid_loss_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.npy'), np.asarray(all_valid_loss))
np.save(os.path.join(save_weights_dir,'all_train_error_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.npy'), np.asarray(all_train_error))
np.save(os.path.join(save_weights_dir,'all_valid_error_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.npy'), np.asarray(all_valid_error))              

plt.figure()
plt.scatter(np.arange(0,len(all_train_loss)), np.asarray(all_train_loss), color='red', alpha=0.5, linewidths=0.8)
plt.scatter(np.arange(0,len(all_valid_loss)), np.asarray(all_valid_loss), color='green', marker='*',alpha=0.5,linewidths=0.8)
plt.show()
plt.savefig(os.path.join(save_weights_dir, 'train_valid_loss_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.png'), dpi=300)

plt.figure()
plt.scatter(np.arange(0,len(all_train_error)), np.asarray(all_train_error), color='red', alpha=0.5, linewidths=0.8)
plt.scatter(np.arange(0,len(all_valid_error)), np.asarray(all_valid_error), color='green', marker='*',alpha=0.5,linewidths=0.8)
plt.show()
plt.savefig(os.path.join(save_weights_dir, 'train_valid_error_overallEpochs'+str(num_epochs)+'_lr'+str(lr)+'.png'), dpi=300)
# '''
""" Test """              
test_loader = data_utils.DataLoader(PAPQMNIST_Bags(train=False,
                                              valid=False,
                                              test=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize(mean=mean_dataset, 
                                                                                       std=std_dataset)
                                    ]), sampling_size=sampling_size_in_instances, 
                                    list_positive_bags = list_posit_bags, 
                                                   data_path=data_path),
                                    batch_size=1,
                                    shuffle=False)
all_names_test = np.load(os.path.join(data_path,'test_imgs_lists.npy'), allow_pickle=True)
    
if args.cuda:
    model = Attention_bags_1GPU(gpu_number, model_architecture)
    model.to('cuda:'+gpu_number)

path = Path(save_weights_dir)


model_name = find('*.pt', save_weights_dir)
if len(model_name)==1:
    new_name_best_inMovAv_val_error_model = model_name[0]
else:
    "Error: Two models saved?" 


model.load_state_dict(torch.load(new_name_best_inMovAv_val_error_model))
subfolder = 'best_inMovAv_val_loss_model'
test(save_weights_dir, subfolder, test_loader, model, test_epochs, args.cuda, gpu_number, all_names_test) 

""" Evaluation """
test_bags = test_bags_info(data_path)    
compute_metrics(subfolder, save_weights_dir, test_bags, data_path, test_epochs) 
