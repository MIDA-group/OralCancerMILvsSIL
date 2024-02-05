import torch
import torch.nn as nn
import torch.nn.functional as F
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import os, fnmatch
import cv2
from PIL import Image
import numpy as np
from pathlib import Path


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

def create_list_of_files_in_given_folder(folder):
    """Lists files in a given directory"""
    list_of_files = []
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            list_of_files.append(os.path.join(folder, file))
    return list_of_files

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def working_dataset(dataset_type, data_path, transforms_0):
    dataset = datasets.ImageFolder(
        os.path.join(data_path, dataset_type),
        transform=transforms_0[dataset_type])
    return dataset

def list_files_oral_cancer3_focused(data_dir, TRAIN, VAL, TEST, negative_class, positive_class):
    negative_class_train_path = os.path.join(data_dir, TRAIN, negative_class)
    positive_class_train_path = os.path.join(data_dir, TRAIN, positive_class)
    negative_class_valid_path = os.path.join(data_dir, VAL, negative_class)
    positive_class_valid_path = os.path.join(data_dir, VAL, positive_class)
    negative_class_test_path = os.path.join(data_dir, TEST, negative_class)
    positive_class_test_path = os.path.join(data_dir, TEST, positive_class)
    
    files_train = list_files_in_folder(negative_class_train_path) + list_files_in_folder(positive_class_train_path)
    files_test = list_files_in_folder(negative_class_test_path) + list_files_in_folder(positive_class_test_path)
    files_valid = list_files_in_folder(negative_class_valid_path) + list_files_in_folder(positive_class_valid_path)

    dict_labels = {}
    for f in list_files_in_folder(negative_class_train_path) + list_files_in_folder(negative_class_test_path)+list_files_in_folder(negative_class_valid_path):
        dict_labels[f] = 0
    for f in list_files_in_folder(positive_class_train_path) + list_files_in_folder(positive_class_test_path)+list_files_in_folder(positive_class_valid_path):
        dict_labels[f] = 1

    return files_train, files_test, files_valid, dict_labels


class AddGaussianNoise(object):
    def __init__(self, mean=0., var1=1., var2=1.):
        var = (var1 - var2) * torch.rand(1) + var2
        self.std = torch.sqrt(var)
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

#def transformations_dataset_oral_cancer3_focused(Train, Val, Test, img_width, img_height, mean_train, std_train):
#    data_transforms = {
#        Train: A.Compose([A.Normalize(mean=mean_train, 
#                                                    std=std_train),
#                                                         A.HorizontalFlip(p=0.5),
#                                                         A.VerticalFlip(p=0.5),
#                                                         A.GaussNoise(var_limit=(2.0, 5.0), p=0.15),
#                                                         A.augmentations.geometric.rotate.RandomRotate90(p=1),                                                     ToTensorV2(),           
#                                     ]),
#        Val: A.Compose([A.Normalize(mean=mean_train, 
#                                                    std=std_train),
#                                                         A.HorizontalFlip(p=0.5),
#                                                         A.VerticalFlip(p=0.5),
#                                                         A.GaussNoise(var_limit=(2.0, 5.0), p=0.15),
#                                                         A.augmentations.geometric.rotate.RandomRotate90(p=1),
#                        ToTensorV2(),           
#                                     ]),
#        Test: A.Compose([A.Normalize(mean=mean_train, std=std_train), ToTensorV2(),])
#    }
#    return data_transforms


def transformations_dataset_oral_cancer3_focused(Train, Val, Test, img_width, img_height, mean_train, std_train):
    data_transforms = {
        Train: transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                         transforms.RandomVerticalFlip(p=0.5),                                 
                                    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                                    transforms.ToTensor(),
                                    transforms.RandomApply(transforms=[AddGaussianNoise(0, var1=2.0, var2=5.0)], p=0.15), 
                                                        transforms.Normalize(mean=mean_train, 
                                                                                        std=std_train),
                                                                   
                                     ]),
        Val: transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                         transforms.RandomVerticalFlip(p=0.5),                                 
                                    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                                    transforms.ToTensor(),
                                    transforms.RandomApply(transforms=[AddGaussianNoise(0, var1=2.0, var2=5.0)], p=0.15), 
                                                        transforms.Normalize(mean=mean_train, 
                                                                                        std=std_train),
                                                                   
                                     ]),
        Test: transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean_train, 
                                                                                        std=std_train)])
    }
    return data_transforms

def load_dataset(dataset, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)
    return loader



def _preprocess(image_path, transformations):
    #image = cv2.imread(image_path)
    #raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #augmented = transformations(image=raw_image)
    raw_image = Image.open(image_path)
    image = transformations(raw_image)
    #image = augmented['image']
    return image, raw_image

def postproc(valid_extr_flag, data_dir, data_transforms, name_class, save_weights_dir, model_path, device, TEST, VAL, model, test_path_bags, list_posit_bags, list_negat_bags,avg_thres):
    perc_as_pos_in_neg_val_bags = [];perc_as_pos_in_pos_val_bags = []
    list_neg_bags_fold=[];list_pos_bags_fold=[]; pred_bag_labels_negat={};pred_bag_labels_posit={}
    if valid_extr_flag:
        test_path = os.path.join(data_dir, 'valid')
        transformations = data_transforms[VAL]

    else:
        test_path = os.path.join(data_dir, 'test')
        transformations = data_transforms[TEST]    
    # Number of images per bag
    num_imgs_in_bags = {'01':15408, '05':3257, '53':3788, '07':6953, '37':11712,'55':1657, '86':12294,'88':3927,'98':4559,'03':36899,'101':8451,'96':2691,'26':17147, '61':15875, '78':2170, '59':26484, '63':1418,'80':22948, '68':26816,'71':10333,'75':14714,'65':44231,'73':7707,'70':6400}

    #num_in_list_posit_bags = {'01':15408, '05':3257, '53':3788, '07':6953, '37':11712,'55':1657, '86':12294,'88':3927,'98':4559,'03':36899,'101':8451,'96':2691}
    #num_in_list_negat_bags = {'26':17147, '61':15875, '78':2170, '59':26484, '63':1418,'80':22948, '68':26816,'71':10333,'75':14714,'65':44231,'73':7707,'70':6400}
    #num_in_list_posit_bags = [15408,3257,3788,6953,11712,1657,12294,3927,4559,36899,8451,2691]
    #num_in_list_negat_bags = [17147,15875,2170,26484,1418,22948,26816,10333,14714,44231,7707,6400]
    TP=0;TN=0;FP=0;FN=0
    num_all_images = 0
    acc_test = 0.0
    batch_size_feed = 20
    for nr in range(len(name_class)): #range(1,2): #
        images_TEM = create_list_of_files_in_given_folder(os.path.join(test_path, name_class[nr]))
        list_of_files = list_files_in_folder(os.path.join(test_path, name_class[nr]))
        num_all_images = num_all_images+len(images_TEM)
        if valid_extr_flag:
            create_save_dir(save_weights_dir, 'valid')
            create_save_dir(os.path.join(save_weights_dir, 'valid'), name_class[nr])
            output_dir = create_save_dir(os.path.join(save_weights_dir, 'valid', name_class[nr]), name_class[nr] + '_'
                                                 + model_path.parts[-1])
        else:
            create_save_dir(save_weights_dir, 'test')
            create_save_dir(os.path.join(save_weights_dir, 'test'), name_class[nr])
            output_dir = create_save_dir(os.path.join(save_weights_dir, 'test', name_class[nr]), name_class[nr] + '_'
                                                 + model_path.parts[-1])        
        # Images class
        print("Images:")
        target_class = nr    
        num_batch = int(np.ceil(len(images_TEM)/batch_size_feed))
        print('num_batch', num_batch)
        for num in range(num_batch):
            images = []
            images_tensor = []
            images_paths = []
            image_names = []

            if num != num_batch-1:
                for i in range(num*batch_size_feed, (num+1)*batch_size_feed):
    #                 print("\t#{}: {}".format(i, images_TEM[i]))
                    images_paths.append(images_TEM[i])
                    image_names.append(list_of_files[i])
                    image, raw_image = _preprocess(images_TEM[i], transformations)
                    images.append(image)
                images_tensor = torch.stack(images).to(device)
                pred1 = F.softmax(model.forward(images_tensor), dim=0)

                labels = torch.LongTensor([target_class] * len(range(num*batch_size_feed, (num+1)*batch_size_feed)))
                _, indices = torch.max(pred1.data, dim=1)

                acc_test += (torch.sum(indices.cpu() == labels.data)).item()

                for ii in range(len(pred1)):
                    np.save(os.path.join(output_dir,Path(images_paths[ii]).parts[-1]+'.npy'), pred1[ii][target_class].cpu().detach().numpy())
                    if labels.data[ii]==1 and indices.cpu()[ii]==1:
                        TP+=1
                    elif labels.data[ii]==0 and indices.cpu()[ii]==0:
                        TN+=1
                    elif labels.data[ii]==0 and indices.cpu()[ii]==1:
                        FP+=1
                    elif labels.data[ii]==1 and indices.cpu()[ii]==0:            
                        FN+=1                

            elif num == num_batch-1:
                for i in range(num * batch_size_feed, len(images_TEM)):
    #                 print("\t#{}: {}".format(i, images_TEM[i]))
                    images_paths.append(images_TEM[i])
                    image_names.append(list_of_files[i])
                    image, raw_image = _preprocess(images_TEM[i], transformations)
                    images.append(image)
                images_tensor = torch.stack(images).to(device)
                pred1 = F.softmax(model.forward(images_tensor),dim=0) 

                labels = torch.LongTensor([target_class] * len(range(num * batch_size_feed, len(images_TEM))))
                _, indices = torch.max(pred1.data, dim=1)

                acc_test += (torch.sum(indices.cpu() == labels.data)).item()
                for ii in range(len(pred1)):
                    np.save(os.path.join(output_dir,Path(images_paths[ii]).parts[-1]+'.npy'), pred1[ii][target_class].cpu().detach().numpy())
                    if labels.data[ii]==1 and indices.cpu()[ii]==1:
                        TP+=1
                    elif labels.data[ii]==0 and indices.cpu()[ii]==0:
                        TN+=1
                    elif labels.data[ii]==0 and indices.cpu()[ii]==1:
                        FP+=1
                    elif labels.data[ii]==1 and indices.cpu()[ii]==0:            
                        FN+=1                                     
        print(name_class[nr], acc_test)
    print(num_all_images)
    print(acc_test/num_all_images)
    Pr = TP/(TP+FP+1e-12)
    Re = TP/(TP+FN+1e-12)
    f1score_test = 2*(Pr*Re)/(Pr+Re+1e-12)
    print('f1score_test', f1score_test)
    
    if valid_extr_flag:
        test_path_ =  os.path.join(Path(test_path_bags).parent, 'valid')
    else:
        test_path_ =  os.path.join(Path(test_path_bags).parent, 'test')

    true_bag_label=[0,1]
    count_TP=0;count_TN=0;count_FP=0;count_FN=0;
    all_true_labels=[]; all_pred_labels=[]
    AUC_all_bags=[]; AUC_positive_bags=[]; sens_all_bags=[]; spec_all_bags=[]
    for cl_ind in range(len(name_class)):
        if cl_ind==0:
            list_bags=list_negat_bags
        else:
            list_bags=list_posit_bags
        if valid_extr_flag:
            output_dir = os.path.join(save_weights_dir, 'valid', name_class[cl_ind], name_class[cl_ind] + '_'+ model_path.parts[-1])
        else:
            output_dir = os.path.join(save_weights_dir, 'test', name_class[cl_ind], name_class[cl_ind] + '_'+ model_path.parts[-1])
        score_names = list_files_in_folder(os.path.join(output_dir))
        for b in range(len(list_bags)):
            all_images_in_bag_score=[]; all_names_in_bag=[]
            all_pred_sample_labels=[]
            bag_folder = os.path.join(test_path_, name_class[cl_ind], list_bags[b])
            if os.path.exists(os.path.join(bag_folder)):
                bag_folder_names = list_files_in_folder(bag_folder)
                for i in range(len(score_names)):
                    score_temp = np.load(os.path.join(output_dir, score_names[i]))
                    if score_names[i][:-4] in bag_folder_names:
                        if score_temp>=0.5:
                            pred_img_label = true_bag_label[cl_ind]
                        elif score_temp<0.5:
                            pred_img_label = np.absolute(true_bag_label[cl_ind]-1)
                        all_pred_sample_labels.append(pred_img_label)
                        all_images_in_bag_score.append(score_temp)
                        all_names_in_bag.append(score_names[i])

                count1s = np.count_nonzero(np.asarray(all_pred_sample_labels).astype(int) == 1)
                count0s = np.count_nonzero(np.asarray(all_pred_sample_labels).astype(int) == 0)
                print('count1s, count0s, bag_folder: ', count1s, count0s, bag_folder)
                #if valid_extr_flag:                
                perc_as_pos_in_a_bag = count1s/num_imgs_in_bags[list_bags[b]]
                if cl_ind==0:
                    list_neg_bags_fold.append(list_bags[b])
                    perc_as_pos_in_neg_val_bags.append(perc_as_pos_in_a_bag)
                else:
                    list_pos_bags_fold.append(list_bags[b])
                    perc_as_pos_in_pos_val_bags.append(perc_as_pos_in_a_bag)
                
                
    if valid_extr_flag:
        mean_perc_as_pos_in_neg_val_bags = np.mean(np.asarray(perc_as_pos_in_neg_val_bags))
        mean_perc_as_pos_in_pos_val_bags = np.mean(np.asarray(perc_as_pos_in_pos_val_bags))
        diffefence = mean_perc_as_pos_in_pos_val_bags-mean_perc_as_pos_in_neg_val_bags
        threshold = diffefence/2+mean_perc_as_pos_in_neg_val_bags
        print('Valid Threshold of the fold: ', threshold)
    else:
        predicted_neg = np.where(np.asarray(perc_as_pos_in_neg_val_bags)<avg_thres, 0,1)
        predicted_pos = np.where(np.asarray(perc_as_pos_in_pos_val_bags)<avg_thres, 0,1)
        
        for_calc_neg = np.where(predicted_neg==0, 1,0)
        for_calc_pos = np.where(predicted_pos==1, 1,0)
        acc = (np.sum(for_calc_neg)+np.sum(for_calc_pos))/(8)
        print('pred labels [neg] [pos]: ', predicted_neg, predicted_pos,'ACC: ', acc)
        for bg in range(len(list_neg_bags_fold)):
            pred_bag_labels_negat[list_neg_bags_fold[bg]] = predicted_neg[bg]
            #pred_bag_labels_negat.update({list_neg_bags_fold[bg]:predicted_neg[bg]})
        for bg in range(len(list_pos_bags_fold)):
            pred_bag_labels_posit[list_pos_bags_fold[bg]] = predicted_pos[bg]
            #pred_bag_labels_posit.update({list_pos_bags_fold[bg]:predicted_pos[bg]})

    return pred_bag_labels_negat, pred_bag_labels_posit


def postproc_new_separate_distrib_pos_neg_bags_train_valid_for_test(set, data_dir, data_transforms, name_class, save_weights_dir, model_path, device, PARTITION, model, test_path_bags, list_posit_bags, list_negat_bags,avg_thres):
    perc_as_pos_in_neg_val_bags = [];perc_as_pos_in_pos_val_bags = []
    list_neg_bags_fold=[];list_pos_bags_fold=[]; pred_bag_labels_negat={};pred_bag_labels_posit={}
    test_path = os.path.join(data_dir, set)
    transformations = data_transforms[PARTITION]
  
    # Number of images per bag
    num_imgs_in_bags = {'01':15408, '05':3257, '53':3788, '07':6953, '37':11712,'55':1657, '86':12294,'88':3927,'98':4559,'03':36899,'101':8451,'96':2691,'26':17147, '61':15875, '78':2170, '59':26484, '63':1418,'80':22948, '68':26816,'71':10333,'75':14714,'65':44231,'73':7707,'70':6400}
    # '''
    #num_in_list_posit_bags = {'01':15408, '05':3257, '53':3788, '07':6953, '37':11712,'55':1657, '86':12294,'88':3927,'98':4559,'03':36899,'101':8451,'96':2691}
    #num_in_list_negat_bags = {'26':17147, '61':15875, '78':2170, '59':26484, '63':1418,'80':22948, '68':26816,'71':10333,'75':14714,'65':44231,'73':7707,'70':6400}
    #num_in_list_posit_bags = [15408,3257,3788,6953,11712,1657,12294,3927,4559,36899,8451,2691]
    #num_in_list_negat_bags = [17147,15875,2170,26484,1418,22948,26816,10333,14714,44231,7707,6400]
    
    TP=0;TN=0;FP=0;FN=0
    num_all_images = 0
    acc_test = 0.0
    batch_size_feed = 20
    for nr in range(len(name_class)): #range(1,2): #
        images_TEM = create_list_of_files_in_given_folder(os.path.join(test_path, name_class[nr]))
        
        list_of_files = list_files_in_folder(os.path.join(test_path, name_class[nr]))
        num_all_images = num_all_images+len(images_TEM)
        if os.path.exists(os.path.join(save_weights_dir, set, name_class[nr])):
            print("Scores already exist, skipping..")


        else:
            create_save_dir(save_weights_dir, set)
            create_save_dir(os.path.join(save_weights_dir, set), name_class[nr])
            output_dir = create_save_dir(os.path.join(save_weights_dir, set, name_class[nr]), name_class[nr] + '_'
                                                    + model_path.parts[-1])
            # Images class
            print("Images:")
            target_class = nr    
            num_batch = int(np.ceil(len(images_TEM)/batch_size_feed))
            print('num_batch', num_batch)
            for num in range(num_batch):
                images = []
                images_tensor = []
                images_paths = []
                image_names = []

                if num != num_batch-1:
                    for i in range(num*batch_size_feed, (num+1)*batch_size_feed):
        #                 print("\t#{}: {}".format(i, images_TEM[i]))
                        images_paths.append(images_TEM[i])
                        image_names.append(list_of_files[i])
                        image, raw_image = _preprocess(images_TEM[i], transformations)
                        images.append(image)
                    images_tensor = torch.stack(images).to(device)
                    # pred1 = F.softmax(model.forward(images_tensor), dim=0)
                    pred1 = F.softmax(model.forward(images_tensor)) 

                    labels = torch.LongTensor([target_class] * len(range(num*batch_size_feed, (num+1)*batch_size_feed)))
                    _, indices = torch.max(pred1.data, dim=1)

                    acc_test += (torch.sum(indices.cpu() == labels.data)).item()

                    for ii in range(len(pred1)):
                        np.save(os.path.join(output_dir,Path(images_paths[ii]).parts[-1]+'.npy'), pred1[ii][target_class].cpu().detach().numpy())
                        if labels.data[ii]==1 and indices.cpu()[ii]==1:
                            TP+=1
                        elif labels.data[ii]==0 and indices.cpu()[ii]==0:
                            TN+=1
                        elif labels.data[ii]==0 and indices.cpu()[ii]==1:
                            FP+=1
                        elif labels.data[ii]==1 and indices.cpu()[ii]==0:            
                            FN+=1                

                elif num == num_batch-1:
                    for i in range(num * batch_size_feed, len(images_TEM)):
        #                 print("\t#{}: {}".format(i, images_TEM[i]))
                        images_paths.append(images_TEM[i])
                        image_names.append(list_of_files[i])
                        image, raw_image = _preprocess(images_TEM[i], transformations)
                        images.append(image)
                    images_tensor = torch.stack(images).to(device)
                    pred1 = F.softmax(model.forward(images_tensor)) 
                    # pred1 = F.softmax(model.forward(images_tensor),dim=0) 

                    labels = torch.LongTensor([target_class] * len(range(num * batch_size_feed, len(images_TEM))))
                    _, indices = torch.max(pred1.data, dim=1)

                    acc_test += (torch.sum(indices.cpu() == labels.data)).item()
                    for ii in range(len(pred1)):
                        np.save(os.path.join(output_dir,Path(images_paths[ii]).parts[-1]+'.npy'), pred1[ii][target_class].cpu().detach().numpy())
                        if labels.data[ii]==1 and indices.cpu()[ii]==1:
                            TP+=1
                        elif labels.data[ii]==0 and indices.cpu()[ii]==0:
                            TN+=1
                        elif labels.data[ii]==0 and indices.cpu()[ii]==1:
                            FP+=1
                        elif labels.data[ii]==1 and indices.cpu()[ii]==0:            
                            FN+=1                                     
            print(name_class[nr], acc_test)
        print(num_all_images)
        print(acc_test/num_all_images)
        Pr = TP/(TP+FP+1e-12)
        Re = TP/(TP+FN+1e-12)
        f1score_test = 2*(Pr*Re)/(Pr+Re+1e-12)
        print('f1score_test', f1score_test)
    # '''
    test_path_ =  os.path.join(Path(test_path_bags).parent, set)
   

    true_bag_label=[0,1]
    # count_TP=0;count_TN=0;count_FP=0;count_FN=0;
    # all_true_labels=[]; all_pred_labels=[]
    # AUC_all_bags=[]; AUC_positive_bags=[]; sens_all_bags=[]; spec_all_bags=[]
    for cl_ind in range(len(name_class)):
        if cl_ind==0:
            list_bags=list_negat_bags
        else:
            list_bags=list_posit_bags
        output_dir = os.path.join(save_weights_dir, set, name_class[cl_ind], name_class[cl_ind] + '_'+ model_path.parts[-1])

        score_names = list_files_in_folder(os.path.join(output_dir))
        for b in range(len(list_bags)):
            all_images_in_bag_score=[]; all_names_in_bag=[]
            all_pred_sample_labels=[]
            bag_folder = os.path.join(test_path_, name_class[cl_ind], list_bags[b])
            if os.path.exists(os.path.join(bag_folder)):
                bag_folder_names = list_files_in_folder(bag_folder)
                for i in range(len(score_names)):
                    score_temp = np.load(os.path.join(output_dir, score_names[i]))
                    if score_names[i][:-4] in bag_folder_names:
                        if score_temp>=0.5:
                            pred_img_label = true_bag_label[cl_ind]
                        elif score_temp<0.5:
                            pred_img_label = np.absolute(true_bag_label[cl_ind]-1)
                        all_pred_sample_labels.append(pred_img_label)
                        all_images_in_bag_score.append(score_temp)
                        all_names_in_bag.append(score_names[i])

                count1s = np.count_nonzero(np.asarray(all_pred_sample_labels).astype(int) == 1)
                count0s = np.count_nonzero(np.asarray(all_pred_sample_labels).astype(int) == 0)
                print('count1s, count0s, bag_folder: ', count1s, count0s, bag_folder)
                #if valid_extr_flag:                
                perc_as_pos_in_a_bag = count1s/num_imgs_in_bags[list_bags[b]]
                if cl_ind==0:
                    list_neg_bags_fold.append(list_bags[b])
                    perc_as_pos_in_neg_val_bags.append(perc_as_pos_in_a_bag)
                else:
                    list_pos_bags_fold.append(list_bags[b])
                    perc_as_pos_in_pos_val_bags.append(perc_as_pos_in_a_bag)
                
                
    if set=='valid' or set=='train':

        # mean_perc_as_pos_in_neg_val_bags = np.mean(np.asarray(perc_as_pos_in_neg_val_bags))
        # mean_perc_as_pos_in_pos_val_bags = np.mean(np.asarray(perc_as_pos_in_pos_val_bags))

        # diffefence = mean_perc_as_pos_in_pos_val_bags-mean_perc_as_pos_in_neg_val_bags
        # threshold = diffefence/2+mean_perc_as_pos_in_neg_val_bags
        # print('Threshold of the fold: ', threshold)
        print(perc_as_pos_in_neg_val_bags, perc_as_pos_in_pos_val_bags)
        return np.asarray(perc_as_pos_in_neg_val_bags), np.asarray(perc_as_pos_in_pos_val_bags), 0
    
    elif set=='test':
        predicted_neg = np.where(np.asarray(perc_as_pos_in_neg_val_bags)<avg_thres, 0,1)
        predicted_pos = np.where(np.asarray(perc_as_pos_in_pos_val_bags)<avg_thres, 0,1)
        
        for_calc_neg = np.where(predicted_neg==0, 1,0)
        for_calc_pos = np.where(predicted_pos==1, 1,0)
        acc_bag = (np.sum(for_calc_neg)+np.sum(for_calc_pos))/(8)
        print('pred labels [neg] [pos]: ', predicted_neg, predicted_pos,'bag ACC: ', acc_bag)
        for bg in range(len(list_neg_bags_fold)):
            pred_bag_labels_negat[list_neg_bags_fold[bg]] = predicted_neg[bg]
            #pred_bag_labels_negat.update({list_neg_bags_fold[bg]:predicted_neg[bg]})
        for bg in range(len(list_pos_bags_fold)):
            pred_bag_labels_posit[list_pos_bags_fold[bg]] = predicted_pos[bg]
            #pred_bag_labels_posit.update({list_pos_bags_fold[bg]:predicted_pos[bg]})

        return pred_bag_labels_negat, pred_bag_labels_posit, acc_bag