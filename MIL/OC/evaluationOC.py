import numpy as np
import os
import sklearn
from sklearn import metrics
import time
from time import perf_counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from functions_OC import list_files_in_folder, create_save_dir
# The code can require changes if not the same format of folder names is used


def compute_metrics(subfolder_valid_approach, weights_folder, bag_names, root, num_samples):
    start_t = perf_counter() 
    count_TP=0;count_TN=0;count_FP=0;count_FN=0; 
    all_true_labels=[]; all_pred_labels=[]
    AUC_all_bags=[]; AUC_positive_bags=[]; 
    true_all_bags_label=[]
    num_test_bags = len(bag_names)
    average_weights_folder = create_save_dir(weights_folder, 'average_weights_'+subfolder_valid_approach)
    print('bag_names', bag_names)
    for bag_num in bag_names: 
        bag_folder = os.path.join(weights_folder, subfolder_valid_approach, str(bag_num).zfill(4))
        list_weights_bag = list_files_in_folder(bag_folder)
        top_number = 36 #50
        if not os.path.exists(os.path.join(root, 'test', 'negative', str(bag_num).zfill(2))):
            bag_folder_test = os.path.join(root, 'test', 'positive', str(bag_num).zfill(2))
            true_bag_label = 1
        else:
            bag_folder_test = os.path.join(root, 'test', 'negative', str(bag_num).zfill(2))
            true_bag_label = 0
            
        # Bag level label
        all_pred_sample_labels=[]
        for s in range(num_samples):
            for i in range(len(list_weights_bag)):
                if list_weights_bag[i][-21:-16]==str(s).zfill(5):
#                 if list_weights_bag[i][-19:-16]==str(s).zfill(2) or list_weights_bag[i][-18:-16]==str(s).zfill(2):
#                     print(list_weights_bag[i][-21:-16])
                    pred_sample_label = list_weights_bag[i][-8:-7]
                    all_pred_sample_labels.append(pred_sample_label)                    
                    break # all instances in sample have the same bag label
                    
        majority_bag_label_from_all_samples_for_one_bag = np.mean(np.asarray(all_pred_sample_labels).astype(int))
        pred_bag_label = 0
        if majority_bag_label_from_all_samples_for_one_bag>=0.5: #majority label bag level
            pred_bag_label=1  
            
        # Bag level metrics
        if true_bag_label==1 and pred_bag_label==1:
            count_TP+=1
        elif true_bag_label==1 and pred_bag_label==0:
            count_FN+=1
        elif true_bag_label==0 and pred_bag_label==1:
            count_FP+=1
        elif true_bag_label==0 and pred_bag_label==0:
            count_TN+=1     
            
        all_true_labels.append(true_bag_label); all_pred_labels.append(pred_bag_label)
#         print('true_bag_label, pred_bag_label', true_bag_label, pred_bag_label) 
        
        # Instance level label
        weights_all_samples_for_one_bag=[]; names_all_samples_for_one_bag=[]
        for s in range(num_samples):
#             print('str(s).zfill(5)', str(s).zfill(5))
            all_inst_weights_in_one_sample=[]; all_inst_names_in_one_sample=[]
            for i in range(len(list_weights_bag)): 
#                 if list_weights_bag[i][-19:-16]==str(s).zfill(2) or list_weights_bag[i][-18:-16]==str(s).zfill(2):
#                 print('list_weights_bag[i][-21:-16]', list_weights_bag[i][-21:-16])
                if list_weights_bag[i][-21:-16]==str(s).zfill(5):
                    coeff = np.load(os.path.join(bag_folder, list_weights_bag[i]))
                    all_inst_weights_in_one_sample.append(coeff)
                    all_inst_names_in_one_sample.append(list_weights_bag[i])
#             print(len(all_inst_weights_in_one_sample))
            min_coef = np.min(np.asarray(all_inst_weights_in_one_sample))
            max_coef = np.max(np.asarray(all_inst_weights_in_one_sample))
            # Normalise weights
            norm_coeff = [(coef-min_coef)/(max_coef-min_coef+10e-12) for coef in all_inst_weights_in_one_sample] 
            weights_all_samples_for_one_bag.append(norm_coeff)
            names_all_samples_for_one_bag.append(all_inst_names_in_one_sample)

        all_test_img_names_in_bag = list_files_in_folder(bag_folder_test)
        average_weights_all_instances_one_bag=[]; majority_label_all_instances_one_bag=[]
        names_test_imgs_sampled_one_bag = []
#         all_images_true_instance_label=[]
        for j in range(len(all_test_img_names_in_bag)):
            one_image_pred_sample_label=[];  one_image_weights=[]
            img_name = all_test_img_names_in_bag[j]  
            for s in range(len(names_all_samples_for_one_bag)):
                temp_names = names_all_samples_for_one_bag[s]
                temp_weights = weights_all_samples_for_one_bag[s]
                for t in range(len(temp_names)):
                    if img_name in temp_names[t]:
                        one_image_weights.append(temp_weights[t])
                        pred_sample_label = temp_names[t][-8:-7]
#                         true_instance_label = temp_names[t][-17:-16]   
                        one_image_pred_sample_label.append(pred_sample_label)
                        
            if not one_image_pred_sample_label:
#                 print('NO EVALUATION FOR IMAGE: ', j)
                names_test_imgs_sampled_one_bag.append('')
            else:
                names_test_imgs_sampled_one_bag.append(img_name)
#             all_images_true_instance_label.append(true_instance_label)
            arr_one_image_pred_sample_label = np.asarray(one_image_pred_sample_label).astype(int)
            count1 = np.count_nonzero(arr_one_image_pred_sample_label == 1)
            if count1>np.around(len(arr_one_image_pred_sample_label)/2):
                majority_pred_samples_label_one_img = 1
            else:
                majority_pred_samples_label_one_img = 0
            count_weights=0; sum_weights=0
            for i in range(len(arr_one_image_pred_sample_label)):
                if arr_one_image_pred_sample_label[i]==majority_pred_samples_label_one_img:
                    sum_weights += one_image_weights[i]
                    count_weights +=1
            average_weight = sum_weights/(count_weights+1e-12)
            average_weights_all_instances_one_bag.append(average_weight)
            majority_label_all_instances_one_bag.append(majority_pred_samples_label_one_img)

        arr = np.asarray(average_weights_all_instances_one_bag)
        maj_arr = np.asarray(majority_label_all_instances_one_bag) #maj. bag level for each instance
        
        print(len(names_test_imgs_sampled_one_bag), len(average_weights_all_instances_one_bag))
        # Save average weights to separate folder
        avg_weights_bag_folder = create_save_dir(average_weights_folder, str(bag_num).zfill(4))
        create_save_dir(avg_weights_bag_folder, 'maj_label_1')
        create_save_dir(avg_weights_bag_folder, 'maj_label_0')
        all_average_weights_one_bag=[]
        all_names_average_weights_one_bag=[]
        for av in range(len(average_weights_all_instances_one_bag)):
            if majority_label_all_instances_one_bag[av] == 1:
                maj_label_folder = os.path.join(avg_weights_bag_folder, 'maj_label_1')
                all_average_weights_one_bag.append(average_weights_all_instances_one_bag[av])
                all_names_average_weights_one_bag.append(names_test_imgs_sampled_one_bag[av])
            elif majority_label_all_instances_one_bag[av] == 0:
                maj_label_folder = os.path.join(avg_weights_bag_folder, 'maj_label_0')
                all_average_weights_one_bag.append(average_weights_all_instances_one_bag[av])
                all_names_average_weights_one_bag.append(names_test_imgs_sampled_one_bag[av])
            else:
                maj_label_folder = os.path.join(avg_weights_bag_folder, 'indefinite_maj_label')
            np.save(os.path.join(maj_label_folder, names_test_imgs_sampled_one_bag[av]+'.npy'), average_weights_all_instances_one_bag[av])
            
        if pred_bag_label==1:  
            inds = np.argsort(np.asarray(all_average_weights_one_bag))
            if len(all_average_weights_one_bag) < top_number:
                top_number = len(all_average_weights_one_bag)
            
            top_weights_img_names = np.asarray(all_names_average_weights_one_bag)[inds[-top_number:]]
            create_save_dir(os.path.join(avg_weights_bag_folder, 'maj_label_1'), 'top')
            np.save(os.path.join(avg_weights_bag_folder, 'maj_label_1', 'top', 'top'+str(top_number)+'.npy'), top_weights_img_names)
        elif pred_bag_label==0:  
            inds = np.argsort(np.asarray(all_average_weights_one_bag))
            if len(all_average_weights_one_bag) < top_number:
                top_number = len(all_average_weights_one_bag)

            top_weights_img_names = np.asarray(all_names_average_weights_one_bag)[inds[-top_number:]]
            create_save_dir(os.path.join(avg_weights_bag_folder, 'maj_label_0'), 'top')
            np.save(os.path.join(avg_weights_bag_folder, 'maj_label_0', 'top', 'top'+str(top_number)+'.npy'), top_weights_img_names) 
            
#     fpr, tpr, thres = sklearn.metrics.roc_curve(np.asarray(all_true_labels), np.asarray(all_pred_labels))
#     AUC_bag_level = sklearn.metrics.auc(fpr, tpr)
    print('Bag true. pred: ', all_true_labels, all_pred_labels)
    acc_bag = np.sum(np.where(np.asarray(all_true_labels)==np.asarray(all_pred_labels), 1, 0))/(num_test_bags+1e-12)
    e = perf_counter() - start_t
    print("Elapsed time during the whole program in seconds:", e)

    print('Accuracy, bag level', "%.3f" % acc_bag)
    print('Bag level confusion matrix', [count_TP,count_FP,count_FN,count_TN])

