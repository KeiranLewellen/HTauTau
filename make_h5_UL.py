from PandaCore.Tools.script import *
from PandaCore.Tools.root_interface import Selector
import PandaCore.Tools.Functions
import numpy as np
import h5py
import os

args = parse('--tag')

tag = args.tag
maxevts = 1300000

dir_dict = {
    "hadhad_UL/"+tag+"/": {
        'QCD_noLep': ['QCD'],
        'WJetsToLNu_noLep': ['WJetsToLNu'],
        'FlatTauTau_user_noLep': ['FlatTauTau_user']
    },
    "hadel_UL/"+tag+"/": {
        'FlatTauTau_user_el': ['FlatTauTau_user'],
        'WJetsToLNu_el': ['WJetsToLNu'],
        'TTbar_el': ['TTbar'],
        'ee-DYJetsToLL_oneEl': ['ee-DYJetsToLL'],
        'tt-DYJetsToLL_el': ['tt-DYJetsToLL']
    },
    "hadmu_UL/" + tag + "/": {
        'FlatTauTau_user_mu': ['FlatTauTau_user'],
        'WJetsToLNu_mu': ['WJetsToLNu'],
        'TTbar_mu': ['TTbar'],
        'mm-DYJetsToLL_oneMu': ['mm-DYJetsToLL'],
        'tt-DYJetsToLL_mu': ['tt-DYJetsToLL']
    },
    "massReg_UL/" + tag + "/": {
        'GluGluHToTauTau_user_noLep': ['GluGluHToTauTau_user'],
        'OtherHToTauTau_user_noLep': ['OtherHToTauTau_user'],
        'tt-DYJetsToLL_noLep': ['tt-DYJetsToLL']
    },
    "massReg_UL_el/" + tag + "/": {
        'GluGluHToTauTau_user_el': ['GluGluHToTauTau_user'],
        'OtherHToTauTau_user_el': ['OtherHToTauTau_user'],
        'tt-DYJetsToLL_el': ['tt-DYJetsToLL']
    },
    "massReg_UL_mu/" + tag + "/": {
        'GluGluHToTauTau_user_mu': ['GluGluHToTauTau_user'],
        'OtherHToTauTau_user_mu': ['OtherHToTauTau_user'],
        'tt-DYJetsToLL_mu': ['tt-DYJetsToLL']
    }
}

inpre = "./"
treeName = 'deepDoubleTau'

for indir in dir_dict:
    if os.path.exists(inpre + indir):
        print(indir)
        for fileSet in dir_dict[indir]:
            files = []
            i = 0
            for file_name in dir_dict[indir][fileSet]:
                while os.path.exists(inpre + indir + file_name + '_' + str(i) + '_w.npy'):
                    files.append(file_name + '_' + str(i))
                    i += 1
                if len(files) > 0:
                    print('converting %s%s%s.z' % (inpre, indir, fileSet))
                    maxevts_per = maxevts / len(files)
                    # Get arrays
                    for fi, fileName in enumerate(files):
                        if fi == 0:
                            np_w = np.load(inpre+indir+fileName+'_w.npy')[:maxevts_per, :]
                            np_x = np.load(inpre+indir+fileName+'_x.npy')[:maxevts_per, :]
                            np_y = np.reshape(np.load(inpre+indir+fileName+'_y.npy'), (-1, 1))[:maxevts_per, :]
                            np_ss = np.load(inpre+indir+fileName+'_ss.npy')[:maxevts_per, :]
                        else:
                            np_w = np.concatenate((np_w, np.load(inpre+indir+fileName+'_w.npy')[:maxevts_per, :]),axis=0)
                            np_x = np.concatenate((np_x, np.load(inpre+indir+fileName+'_x.npy')[:maxevts_per, :]),axis=0)
                            np_y = np.concatenate((np_y, np.reshape(np.load(inpre+indir+fileName+'_y.npy'), (-1, 1))[:maxevts_per, :]), axis=0)
                            np_ss = np.concatenate((np_ss, np.load(inpre+indir+fileName+'_ss.npy')[:maxevts_per, :]), axis=0)
                        print(np_w.shape)
                        print(np_ss.shape)
                        print(np_x.shape)
                        print(np_y.shape)
                        if len(np_y)>maxevts:
                            break
                    np_w = np_w[:maxevts, :]
                    np_x = np_x[:maxevts, :]
                    np_y = np_y[:maxevts, :]
                    np_ss = np_ss[:maxevts, :]
                    arr = np.concatenate((np_w, np_ss, np_x, np_y), axis=1)
                    print(arr.shape)
                    # open HDF5 file and write dataset
                    h5File = h5py.File(indir+fileSet+'.z', 'w')
                    h5File.create_dataset(treeName, data=arr,  compression='lzf')
                    h5File.close()
                    del h5File

