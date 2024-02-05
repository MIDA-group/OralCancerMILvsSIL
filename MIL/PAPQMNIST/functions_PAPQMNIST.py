import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.squeezenet import SqueezeNet, Fire
from torchvision.models.resnet import ResNet, Bottleneck
from PIL import Image
from pathlib import Path
import numpy as np
import os, fnmatch


def list_files_in_folder(image_folder):
    """Lists file names in a given directory"""
    list_of_files = []
    for file in os.listdir(image_folder):
        if os.path.isfile(os.path.join(image_folder, file)):
            list_of_files.append(file)
    return list_of_files

def create_save_dir(direct, name_subdirectory):
    if not os.path.exists(os.path.join(direct, name_subdirectory)):
        print('make dir')
        os.mkdir(os.path.join(direct, name_subdirectory))
    return os.path.join(direct, name_subdirectory)
    

class PAPQMNIST_Bags(data_utils.Dataset):
    def __init__(self, transform, sampling_size, list_positive_bags, train, valid, test, data_path):
        self.train = train; self.valid = valid; self.test = test
        self.transform = transform
        self.sampling_size = sampling_size
        self.list_positive_bags = list_positive_bags
        if self.train==True:
            self.datapath = os.path.join(data_path, 'train')
            self.list_paths = self._list_bags_paths()
            self.train_bags_list, self.train_labels_list, self.train_imgs_lists, self.bags_names_train = self._create_bags()
            np.save(os.path.join(data_path, 'train_imgs_lists.npy'), np.asarray(self.train_imgs_lists))
        elif self.valid==True:
            self.datapath = os.path.join(data_path, 'valid')
            self.list_paths = self._list_bags_paths()
            self.valid_bags_list, self.valid_labels_list, self.valid_imgs_lists, self.bags_names_valid = self._create_bags()
            np.save(os.path.join(data_path, 'valid_imgs_lists.npy'), np.asarray(self.valid_imgs_lists))            
        elif self.test==True:
            self.datapath = os.path.join(data_path, 'test')
            self.list_paths = self._list_bags_paths()
            self.test_bags_list, self.test_labels_list, self.test_imgs_lists, self.bags_names_test = self._create_bags()
            np.save(os.path.join(data_path,'test_imgs_lists.npy'), np.asarray(self.test_imgs_lists, dtype=object))

    def _list_bags_paths(self):
        list_paths = []
        basePath = Path(self.datapath)
        for child in basePath.iterdir():
            if child.is_dir():
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        list_paths.append(grandchild)
        return list_paths
    
    def _create_bags(self):
        all_bags = [] 
        all_bags_labels = [] 
        lists_imgs_all = []
        bags_names_all=[]

        for p in range(len(self.list_paths)):
            list_img_names = list_files_in_folder(str(self.list_paths[p]))
            basePath = Path(self.list_paths[p])
            bags_names_all.append(np.asarray(int(basePath.parts[-1])))
            print(basePath.parts[-3], basePath.parts[-1])
            list_img_names_bag = ()
            
            for i in range(len(list_img_names)):
                temp = (list_img_names[i],)
                list_img_names_bag = list_img_names_bag+temp
            lists_imgs_all.append(np.asarray(list_img_names_bag))

            images = [self.transform(Image.open(os.path.join(str(self.list_paths[p]), list_img_names_bag[i]))).float() for i in range(len(list_img_names_bag))]
            images = torch.stack(images)
            X = images
#             X = torch.empty(1, 3, 80, 80)
            y = torch.empty(1)
            for i in range(len(list_img_names)):
#                 img = Image.open(os.path.join(str(self.list_paths[p]), list_img_names[i]))
#                 img_tensor = self.transform(img).float()
#                 X = torch.cat((X,torch.unsqueeze(img_tensor, 0)),0)
                if list_img_names[i][0]=='4':
                    y = torch.cat((y,torch.ones(1)),0)
                else:
                    y = torch.cat((y,torch.zeros(1)),0)

            all_bags.append(X) #(X[1:,:,:,:]) #(X) 
            all_bags_labels.append(y[1:])

        return all_bags, all_bags_labels, lists_imgs_all, bags_names_all 
    
    def __len__(self):
        return len(self.list_paths)

    def __getitem__(self, index):
        # Introduce sampling, with replacement for train, test and valid
        max_number_of_imgs_in_bag = int(self.sampling_size)
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
            lists_of_names = self.train_imgs_lists[index]
            bages_names = self.bags_names_train[index]
        elif self.valid:
            bag = self.valid_bags_list[index]
            label = [max(self.valid_labels_list[index]), self.valid_labels_list[index]]
            lists_of_names = self.valid_imgs_lists[index]
            bages_names = self.bags_names_valid[index]            
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
            lists_of_names = self.test_imgs_lists[index]
            bages_names = self.bags_names_test[index]

        if bag.shape[0] > max_number_of_imgs_in_bag: 
            sample_indices = torch.randint(bag.shape[0], (max_number_of_imgs_in_bag,))
            bag_sampled = bag[sample_indices,:,:,:]
            label_sampled = [label[0],label[1][sample_indices]]
            lists_of_names_sampled = [lists_of_names[x] for x in sample_indices]
            bages_names_sampled = bages_names       
        else:
            sample_indices = np.arange(bag.shape[0])
            bag_sampled = bag
            label_sampled = label
            lists_of_names_sampled = lists_of_names
            bages_names_sampled = bages_names     
        
        return bag_sampled, label_sampled, torch.as_tensor(sample_indices), torch.as_tensor(index), torch.as_tensor(bages_names_sampled)
    
    


class AddGaussianNoise(object):
    def __init__(self, mean=0., var1=1., var2=1.):
        var = (var1 - var2) * torch.rand(1) + var2
        self.std = torch.sqrt(var)
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
def count_max_num_instances(datapath):
    count_max = 0
    basePath = Path(datapath)
    for child in basePath.iterdir():
        if child.is_dir() and (child.parts[-1]=='train' or child.parts[-1]=='valid' or child.parts[-1]=='test'):
            for grandchild in child.iterdir():
                if grandchild.is_dir():
                    for ggrandchild in grandchild.iterdir():
                        if ggrandchild.is_dir():
                            count_tmp = len(list_files_in_folder(ggrandchild))
                            if count_tmp>=count_max:
                                count_max=count_tmp
    return count_max


def test_bags_info(data_path):
    test_bags = []
    basePath = Path(data_path)
    for child in basePath.iterdir():
        if child.is_dir() and child.parts[-1]=='test':
            for grandchild in child.iterdir():
                if grandchild.is_dir():
                    for ggrandchild in grandchild.iterdir():
                        if ggrandchild.is_dir():                
                            test_bags.append(ggrandchild.parts[-1])
    return test_bags

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def test(save_weights_dir, subfolder, test_loader, model, test_epochs, cuda, gpu_number, all_names_test):
    model.eval()
    # test_loss = 0.; test_error = 0.
    for epoch in range(0, test_epochs): 
        print('test_epoch', epoch, end='\r')
        for batch_idx, (data, label, sample_indices, index, bages_names) in enumerate(test_loader):
            bag_label = label[0]
            instance_labels = label[1] 
            if cuda:
                data, bag_label = data.to('cuda:'+gpu_number), bag_label.to('cuda:'+gpu_number) 
            data, bag_label = Variable(data), Variable(bag_label)
            loss, attention_weights = model.calculate_objective(data, bag_label)
            # test_loss += loss.data[0]
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            # test_error += error
            
            create_save_dir(save_weights_dir, subfolder)
            save_weights_bag = create_save_dir(os.path.join(save_weights_dir, subfolder), \
                                               str(bages_names[0].cpu().numpy()).zfill(4))
            # print(len(attention_weights.cpu().data), attention_weights.cpu().data.numpy()[0])                                  
            for i in range(data.cpu().shape[1]):
                label_name = int(label[1][0][i].cpu().numpy())
                # Save attention weights
                np.save(os.path.join(save_weights_bag, str(all_names_test[index][sample_indices[0][i]])+'_'+\
                                     str(epoch).zfill(5)+'_'+str(label_name)+'_'+\
                                     str(bages_names[0].cpu().numpy()).zfill(4)+'_'+\
                                     str(predicted_label.cpu().numpy())+".npy"), \
                        attention_weights.cpu().data.numpy()[0][i]) 
                
                