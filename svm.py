from sklearn.utils import resample
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import optunity
import optunity.metrics

data_path = 'data/brainfc'

label = 'gender_label.npy'

files = ['brainfc_-0.1.npy','brainfc_-0.2.npy','brainfc_-0.3.npy','brainfc_-0.4.npy','brainfc_0.0.npy','brainfc_0.1.npy','brainfc_0.2.npy','brainfc_0.3.npy','brainfc_0.4.npy','brainfc_0.5.npy']

gender_label = np.load(Path(data_path)/Path(label), allow_pickle=True)

kf = KFold(5, shuffle=True, random_state=1)
splited_datas = kf.split(gender_label)

for f in files:
    fc_data = np.load(Path('saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999')/Path(f))
    scores = []
    splited_datas = kf.split(gender_label)
    for train_index, test_index in splited_datas:


        train_label, test_label = gender_label[train_index], gender_label[test_index]


        svm = SVC(gamma='auto').fit(fc_data[train_index], train_label)

        pre = svm.predict(fc_data[test_index])

        score = roc_auc_score(test_label, pre) 
        scores.append(score)
    print(np.mean(np.array(scores), axis=0))







index_list = ['Somatomotor Hand', 'Somatomotor Mouth',
              'Cerebellar', 'Subcortical', 'Visual', 'Auditory',
              'Cingulo-opercular', 'Default mode', 'Fronto-parietal', 'Ventral attention', 'Salience',
              'Dorsal attention', 'Memory retrieval']


data_path = 'data/brainfc'

label = 'gender_label.npy'
fc_data = 'sub_network.npy'

fc_data = np.load(Path('saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999')/Path('sub_brainfc_0.2.npy'), allow_pickle=True)

index = np.load(Path(data_path)/Path('index_subnetwork.npy'), allow_pickle=True)

gender_label = np.load(Path(data_path)/Path(label), allow_pickle=True)

module_names = set([i[1] for i in index])
length = fc_data.shape[1]

Xs = {}

for name in module_names:
    Xs[name] = np.zeros((503, length))

for fc, idx in zip(fc_data, index):
    Xs[idx[1]][int(idx[0])] = fc

kf = KFold(5, shuffle=True, random_state=1)
splited_datas = kf.split(gender_label)
folders = []
Xs_train = []
Xs_test = []




# score function: twice iterated 10-fold cross-validated accuracy

coef = {'Dorsal attention': (0.152277843806741, 8.413194659895563e-05), 'Default mode': (0.002832614669903672, 0.008899692474029004), 'Visual': (0.0012585291561966205, 0.010528895431979018), 'Somatomotor Hand': (91.32680568494469, 3.0796201330239446), 'Memory retrieval': (0.01718984287776791, 0.0008233684337611531), 'Somatomotor Mouth': (0.004462233684622663, 0.09562755705101463), 'Ventral attention': (2.374580366939407, 0.008308454566110589), 'Salience': (0.27590636451282075, 0.0014582033643190928), 'Auditory': (1.1984316810057112e-05, 0.000568699455581053), 'Cerebellar': (0.07953376425893045, 1.3377345730825068), 'Subcortical': (0.6525062922679831, 0.0782923310603226)}

for name, X in Xs.items():

    data = Xs[name]
    labels = gender_label
    

    @optunity.cross_validated(x=data, y=labels, num_folds=10, num_iter=2)
    def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
        model = SVC(C=10 ** logC, gamma=10 ** logGamma, probability=True).fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        return optunity.metrics.roc_auc(y_test, decision_values)

    # perform tuning
    hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])

    # train model on the full training set with tuned hyperparameters
    optimal_model = SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(data, labels)
    coef[name] = (10 ** hps['logC'], 10 ** hps['logGamma'])

print(coef)

for train_index, test_index in splited_datas:


    train_label, test_label = gender_label[train_index], gender_label[test_index]
    estimators = []

    for name, X in Xs.items(): 

        svm = SVC(C=coef[name][0], gamma=coef[name][1], probability=True).fit(X[train_index], train_label)
        estimators.append(svm)
    preds = []
    for e in estimators:
        pre = e.predict(X[test_index])
        preds.append(pre)

    preds = np.array(preds).T
    preds = np.average(preds, axis=1)

    score = roc_auc_score(test_label, preds)
    print(score)



    # svm_bag = []
    # preds = []

    # for x_train, x_test in zip(Xs_train, Xs_test):
    #     svm = SVC(gamma='auto',probability=True)
    #     svm.fit(x_train, train_label)
    #     ans = svm.predict(x_test)
    #     preds.append(ans)

    # preds = np.array(preds)






# print("C": )








