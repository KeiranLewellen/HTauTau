# Imports basics

import numpy as np
import pandas as pd
import h5py
import json

# Opens json files for signal and background

with open("../pf.json") as jsonfile:
    payload = json.load(jsonfile)
    weight = payload['weight']
    features = payload['features']
    altfeatures = payload['altfeatures']
    cut = payload['cut']
    ss = payload['ss_vars']
    label = payload['!decayType']

with open("../pf.json") as jsonfile:
    payload_bkg = json.load(jsonfile)
    weight_bkg = payload_bkg['weight']
    features_bkg = payload_bkg['features']
    altfeatures_bkg = payload_bkg['altfeatures']
    cut_bkg = payload_bkg['cut']
    ss_bkg = payload_bkg['ss_vars']
    label_bkg = payload_bkg['!decayType']

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
lColumns = lColumns + lPartfeatures + lSVfeatures + [label]

# Creates the column names of the background data frame

lColumns_bkg = weight_bkg + ss_bkg
nparts = 30
lPartfeatures_bkg = []
for iVar in features_bkg:
    for i0 in range(nparts):
        lPartfeatures_bkg.append(iVar + str(i0))
nsvs = 5
lSVfeatures_bkg = []
for iVar in altfeatures_bkg:
    for i0 in range(nsvs):
        lSVfeatures_bkg.append(iVar + str(i0))
lColumns_bkg = lColumns_bkg + lPartfeatures_bkg + lSVfeatures_bkg + [label_bkg]

# Creates the column names of the end data frame

lColumns_end = weight + ss
nparts = 30
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

nsvs = 5
lSVfeatures_end = []
for iVar in altfeatures:
    for i0 in range(nsvs):
        lSVfeatures_end.append(iVar + str(i0))
lColumns_end = lColumns_end + lPartfeatures_end + lSVfeatures_end + [label]

# Defines fill factor for each type of background

fill_factor = 2

# Makes a data set where the distribution of the background across mass and pT is similar to that of the signal

def remake(iFiles_sig, iFiles_bkg, iFile_out):
    '''remake(list[array(nxm),...], list[array(nxs),...], str)'''

    # Creates the signal data frame

    features_labels_df_sig = pd.DataFrame(columns=lColumns)

    for sig in iFiles_sig:
        h5File_sig = h5py.File(sig)
        treeArray_sig = h5File_sig['deepDoubleTau'][()]
        print(treeArray_sig.shape)
        tmp_sig = pd.DataFrame(treeArray_sig, columns=lColumns)
        features_labels_df_sig = pd.concat([features_labels_df_sig, tmp_sig])

    # Calculates the distribution of the signal

    sighist, _x, _y = np.histogram2d(features_labels_df_sig[weight[0]], features_labels_df_sig[weight[1]], bins=20,
                                     range=np.array([[200., 800.], [40., 240.]]))
    print(np.sum(sighist))

    # Creates the background data frame

    remade_df_bkg = pd.DataFrame(columns=lColumns_bkg)
    for bkg in iFiles_bkg:
        h5File_bkg = h5py.File(bkg)
        treeArray_bkg = h5File_bkg['deepDoubleTau'][()]
        print(treeArray_bkg.shape)
        tmp_bkg = pd.DataFrame(treeArray_bkg, columns=lColumns_bkg)

        # Adds background based on signal distribution until fill factor is reached

        for ix in range(len(_x) - 1):
            print(len(_x))
            for iy in range(len(_y) - 1):
                remade_df_bkg = pd.concat([remade_df_bkg, tmp_bkg[(
                            (tmp_bkg[weight[0]] >= _x[ix]) & (tmp_bkg[weight[0]] < _x[ix + 1]) & (
                                tmp_bkg[weight[1]] >= _y[iy]) & (tmp_bkg[weight[1]] < _y[iy + 1]))].head(
                    int(int(sighist[ix, iy]) * fill_factor))], ignore_index=True)

    # Shows fill factor per bin

    bkghist, _, _ = np.histogram2d(remade_df_bkg[weight[0]], remade_df_bkg[weight[1]], bins=20,
                                   range=np.array([[200., 800.], [40., 240.]]))
    print(np.nan_to_num(np.divide(bkghist, sighist)))

    # Merges data frames

    merged_df = features_labels_df_sig.append(remade_df_bkg)

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

    temp_columns = pd.DataFrame(temp_columns,columns=merged_df.columns)

    merged_df = merged_df.append(temp_columns)

    oneHot_df = pd.get_dummies(merged_df, prefix=columns, columns=columns)

    oneHot_df = oneHot_df.iloc[:-11]

    # Creates output file

    merged_df = oneHot_df[lColumns_end]
    final_df = merged_df[~(np.sum(np.isinf(merged_df.values), axis=1) > 0)]
    print(list(final_df.columns))
    arr = final_df.values
    print(arr.shape)

    # Open HDF5 file and write dataset

    h5File = h5py.File(iFile_out, 'w')
    h5File.create_dataset('deepDoubleTau', data=arr, compression='lzf')
    h5File.close()
    del h5File


if __name__ == "__main__":
    # remake(['./GluGluHToTauTau_user_hadel.z'],['./WJets.z','./TTbar.z'],'./comb_distcut%i_hadel.z'%fill_factor)
    # remake(['./GluGluHToTauTau_user_hadmu.z'],['./WJets.z','./TTbar.z'],'./comb_distcut%i_hadmu.z'%fill_factor)
    remake(['/data/t3home000/keiran/HtautauData_200GeV/hadhad/FlatTauTau_user_0.z'], ['/data/t3home000/keiran/HtautauData_200GeV/QCD_0.z'], '/data/t3home000/keiran/HtautauData_200GeV/hadhad/comb_distcut%i_flat_hadhad_QCD_0_eho.z' % fill_factor)
