# Imports basics

import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt

pt_range = [200., 800.]
mass_range = [10., 400.]
fill_factor = [0.75, 0.5]
signal_list = ['FlatTauTau_user_noLep.z']
background_list = ['QCD_noLep.z', 'WJetsToLNu_noLep.z']
output_name = "comb_distcut1:0_75:0_5_flat_hadhad_QCD_WJets_noLep,ohe,taus,metCut40,UL,multiclass.z"
base = "/nobackup/users/keiran/training_data/"
tag = "hadhad_UL/Jan17/"
met_cut = 40
diagram_name = "hadhad sigShaped"


# Opens json files for signal and background

base_dir = base + tag
with open(base + "pf_allData_UL.json") as jsonfile:
    payload = json.load(jsonfile)
    weight = payload['weight']
    features = payload['features']
    altfeatures = payload['altfeatures']
    taufeatures = payload['taufeatures']
    elecfeatures = payload['elecfeatures']
    muonfeatures = payload['muonfeatures']
    cut = payload['cut']
    ss = payload['ss_vars']
    label = payload['!decayType']

# Creates the column names of the signal data frame

lColumns = weight + ss
nparts = 30
lPartfeatures = []
for iVar in features:
    for i0 in range(nparts):
        lPartfeatures.append(iVar + str(i0))
nsvs = 5
lSVfeatures = []
for iVar in altfeatures:
    for i0 in range(nsvs):
        lSVfeatures.append(iVar + str(i0))
ntaus = 3
lTaufeatures = []
for iVar in taufeatures:
    for i0 in range(ntaus):
        lTaufeatures.append(iVar + str(i0))
nelecs = 2
lElecfeatures = []
for iVar in elecfeatures:
    for i0 in range(nelecs):
        lElecfeatures.append(iVar + str(i0))
nmuons = 2
lMuonfeatures = []
for iVar in muonfeatures:
    for i0 in range(nmuons):
        lMuonfeatures.append(iVar + str(i0))

lColumns = weight + ss + lPartfeatures + lSVfeatures + lTaufeatures + lElecfeatures + lMuonfeatures

# Creates the column names of the end data frame

lPartfeatures_end = []
PF_id_index = features.index("PF_id")
for k in range(PF_id_index):
    iVar = features[k]
    for i0 in range(nparts):
        lPartfeatures_end.append(iVar + str(i0))

for i in range(11):
    for j in range(nparts):
        lPartfeatures_end.append("PF_id"+str(j)+'_'+str(i))

for k in range(PF_id_index+1, len(features)):
    iVar = features[k]
    for i0 in range(nparts):
        lPartfeatures_end.append(iVar + str(i0))

lColumns_end = weight + ss + lPartfeatures_end + lSVfeatures + lTaufeatures + lElecfeatures + lMuonfeatures

# Makes a data set where the distribution of the background across mass and pT is similar to that of the signal

def remake(iFiles_sig, iFiles_bkg, fill_factor, iFile_out):
    '''remake(list[array(nxm),...], list[array(nxs),...], list, str)'''

    for i in range(len(iFiles_sig)+len(iFiles_bkg)):
        lColumns_end.append("label"+str(i))

    # Creates the signal data frame

    for i in range(len(iFiles_sig)):
        h5File_sig = h5py.File(base_dir + iFiles_sig[i])
        treeArray_sig = h5File_sig['deepDoubleTau'][()]
        print(treeArray_sig.shape)
        features_labels_df_sig = pd.DataFrame(treeArray_sig[:, 0:-1], columns=lColumns).astype('float32')
        treeArray_sig = 0
        features_labels_df_sig["label" + str(i)] = 1
        pt_col = features_labels_df_sig[weight[0]].values.reshape(-1, 1)
        mass_col = features_labels_df_sig[weight[1]].values.reshape(-1, 1)
        features_labels_df_sig = features_labels_df_sig[
            np.logical_and(np.logical_and(np.greater(pt_col, pt_range[0]), np.less(pt_col, pt_range[1])),
                           np.logical_and(np.greater(mass_col, mass_range[0]), np.less(mass_col, mass_range[1])))]
        features_labels_df_sig = features_labels_df_sig[(features_labels_df_sig["MET_pt"] > met_cut)]

    # Calculates the distribution of the signal

    sighist, _x, _y = np.histogram2d(features_labels_df_sig[weight[0]], features_labels_df_sig[weight[1]], bins=20,
                                     range=np.array([pt_range, mass_range]))
    print(np.sum(sighist))
    # Creates the background data frame

    remade_df_bkg = pd.DataFrame(columns=lColumns)
    for i in range(len(iFiles_bkg)):
        h5File_bkg = h5py.File(base_dir + iFiles_bkg[i])
        treeArray_bkg = h5File_bkg['deepDoubleTau'][()]
        print(treeArray_bkg.shape)
        tmp_bkg = pd.DataFrame(treeArray_bkg[:, 0:-1], columns=lColumns).astype('float32')
        treeArray_bkg = 0
        tmp_bkg["label" + str(i+len(iFiles_sig))] = 1
        tmp_bkg = tmp_bkg[(tmp_bkg["MET_pt"] > met_cut)]
        print(tmp_bkg.shape)

        # Adds background based on signal distribution until fill factor is reached

        for ix in range(len(_x) - 1):
            print(len(_x))
            for iy in range(len(_y) - 1):
                remade_df_bkg = pd.concat([remade_df_bkg, tmp_bkg[(
                            (tmp_bkg[weight[0]] >= _x[ix]) & (tmp_bkg[weight[0]] < _x[ix + 1]) & (
                                tmp_bkg[weight[1]] >= _y[iy]) & (tmp_bkg[weight[1]] < _y[iy + 1]))].head(
                    int(int(sighist[ix, iy]) * fill_factor[i]))], ignore_index=True)
        tmp_bkg = 0
    
    # Calculates the distribution of the shaped background
    
    bkghist, _, _ = np.histogram2d(remade_df_bkg[weight[0]], remade_df_bkg[weight[1]], bins=20,
                                   range=np.array([pt_range, mass_range]))
    print(np.sum(bkghist))

    # Creates the shaped signal data frame

    lColumns_sig = lColumns
    for i in range(len(iFiles_sig)):
        lColumns_sig = lColumns.append("label" + str(i))
    remade_df_sig = pd.DataFrame(columns=lColumns)

    # Adds signal based on background distribution until fill factor is reached
    
    fill_factor_tot = 0
    for i in fill_factor:
        fill_factor_tot += i
        
    for ix in range(len(_x) - 1):
        print(len(_x))
        for iy in range(len(_y) - 1):
            remade_df_sig = pd.concat([remade_df_sig, features_labels_df_sig[(
                (features_labels_df_sig[weight[0]] >= _x[ix]) & (features_labels_df_sig[weight[0]] < _x[ix + 1]) & 
                (features_labels_df_sig[weight[1]] >= _y[iy]) & (features_labels_df_sig[weight[1]] < _y[iy + 1]))].head(
                int(int(bkghist[ix, iy]) * (1/fill_factor_tot)))], ignore_index=True)
    features_labels_df_sig = 0    
    
    # Calculates the distribution of the shaped signal
    
    sighist_remade, _x, _y = np.histogram2d(remade_df_sig[weight[0]], remade_df_sig[weight[1]], bins=20,
                                     range=np.array([pt_range, mass_range]))
    print(np.sum(sighist_remade))

    # Shows fill factor per bin

    div_hist = np.nan_to_num(np.divide(bkghist, sighist_remade))
    div_hist = div_hist.astype("float32")
    print(div_hist)
    
    # Creates and saves figure
    
    title = "Training data bkg:sig: " + str(diagram_name)
    plt.figure()
    edges = pt_range + mass_range
    plt.imshow(div_hist.T,extent=edges,origin='low',aspect='auto')
    plt.colorbar()
    plt.ylabel("Jet pT")
    plt.xlabel("Jet mass")
    plt.title(title)
    plt.savefig(title)
    
    # Merges data frames

    merged_df = pd.concat([remade_df_sig, remade_df_bkg]).astype('float32')
    merged_df = merged_df.fillna(0)

    # Relabels particle IDs

    idconv = {211.: 1, 13.: 2, 22.: 3, 11.: 4, 130.: 5, 1.: 6, 2.: 7, 3.: 8, 4.: 9,
              5.: 10, -211.: 1, -13.: 2,
              -11.: 4, -1.: -6, -2.: 7, -3.: 8, -4.: 9, -5.: 10, 0.: 0}

    for i0 in range(nparts):
        merged_df['PF_id' + str(i0)] = merged_df['PF_id' + str(i0)].map(idconv)

    # One hot encodes particle data

    columns = []
    for i0 in range(nparts):
        columns.append('PF_id' + str(i0))

    temp_columns=[]
    for i in range(11):
        column = []
        for col in merged_df.columns:
            if col in columns:
                column.append(i)
            else:
                column.append(np.nan)
        temp_columns.append(column)

    temp_columns = pd.DataFrame(temp_columns, columns=merged_df.columns)

    merged_df = merged_df.append(temp_columns)

    oneHot_df = pd.get_dummies(merged_df, prefix=columns, columns=columns)

    oneHot_df = oneHot_df.iloc[:-11]

    # Creates output file
    merged_df = oneHot_df[lColumns_end]
    oneHot_df = 0
    final_df = merged_df[~(np.sum(np.isinf(merged_df.values), axis=1) > 0)].astype('float32')
    merged_df = 0
    print(list(final_df.columns))
    print(final_df.values.shape)

    # Open HDF5 file and write dataset

    h5File = h5py.File(base_dir + iFile_out, 'w')
    h5File.create_dataset('deepDoubleTau', data=final_df.values, compression='lzf')
    h5File.close()
    del h5File


remake(signal_list, background_list, fill_factor, output_name)
