#!/usr/bin/env python

from PandaCore.Tools.script import * 
from PandaCore.Tools.root_interface import Selector 
import PandaCore.Tools.Functions
import numpy as np
import json 
import os

args = parse('--out', '--name', '--json', '--tag')
startstep = 0
stepsize = 500000

try:
    os.makedirs(args.out)
except OSError:
    pass

sample_dict = {
    "QCD":0,
    "TTbar":0,
    "WJetsToLNu":0,
    "DYJetsToLL":1,
    "ee-DYJetsToLL":0,
    "mm-DYJetsToLL":0,
    "tt-DYJetsToLL":1,
    "JetHT":0,
    "JetHT-2017B-lepveto":0,
    "JetHT-2017C-lepveto":0,
    "JetHT-2017D-lepveto":0,
    "JetHT-2017E-lepveto":0,
    "JetHT-2017F-lepveto":0,
    "JetHT-2017B":0,
    "JetHT-2017C":0,
    "JetHT-2017D":0,
    "JetHT-2017E":0,
    "JetHT-2017F":0,
    "GluGluHToTauTau_user":1,
    "OtherHToTauTau_user":1,
    "FlatTauTau_user":1,
}

def save(arr, label, it):
    fout = args.out+'/'+args.name+'_'+str(it)+'_'+label+'.npy'
    np.save(fout, arr)
    logger.info(sys.argv[0], 'Saved to '+fout)

with open(args.json) as jsonfile:
    payload = json.load(jsonfile)
    weight = payload['weight']
    basedir = payload['base']
    features = payload['features']
    altfeatures = payload['altfeatures']
    taufeatures = payload['taufeatures']
    elecfeatures = payload['elecfeatures']
    muonfeatures = payload['muonfeatures']
    cut = payload['cut']
    ss = payload['ss_vars']
    for i,s in enumerate(payload['samples']):
        if s['name'] == args.name:
            samples = s['samples']
            y = sample_dict[args.name]
            break
    if (not samples):
        logger.error(sys.argv[0], 'Could not identify process '+args.name)
        sys.exit(1)


print(args.name)
if (y!=0): 
    if ("hadhad" in args.out):  cut+=" && fj_nProngs == 2 && fj_nEles == 0 && fj_nMus == 0"
    elif ("hadel" in args.out): cut+=" && fj_nProngs == 2 && fj_nEles == 1 && fj_nMus == 0"
    elif ("hadmu" in args.out): cut+=" && fj_nProngs == 2 && fj_nEles == 0 && fj_nMus == 1"
elif ("-DYJetsToLL" in args.name):
    cut+=" && fj_nProngs == 2"
print(cut)

s = Selector()
chain = root.TChain('Events')
for sample in samples:
    chain.AddFile(basedir + '/' + sample + "-" + args.tag + '/out.root')
logger.info(sys.argv[0], 'Reading files for process '+args.name)
print("%i total events..."%chain.GetEntries())
for si,slim in enumerate(range(startstep*stepsize,chain.GetEntries(),stepsize)):
    selargs = {"start":slim, "stop":slim+stepsize, "step":1}
    print("\tRunning iteration %i"%(si+startstep))
    #if ("JetHT" in args.name): selargs = {}
    s.read_tree(chain, branches=(features+altfeatures+taufeatures+elecfeatures+muonfeatures+weight+ss), cut=cut, xkwargs=selargs)
    print(s["PF_pt"][:,0])
    X = np.concatenate([np.vstack([s[f].T for f in features]).T,np.vstack([s[f].T for f in altfeatures]).T,np.vstack([s[f].T for f in taufeatures]).T,np.vstack([s[f].T for f in elecfeatures]).T,np.vstack([s[f].T for f in muonfeatures]).T],axis=1)
    W = np.vstack([s[var] for var in weight]).T
    Y = y * np.ones(shape=W.shape[0])
    print(Y)
    ss_arr = np.vstack([s[var] for var in ss]).T
    print(X.shape)
    print(Y.shape)
    print(W.shape)
    print(ss_arr.shape)
    
    save(X, 'x',si+startstep)
    save(Y, 'y',si+startstep)
    save(W, 'w',si+startstep)
    save(ss_arr, 'ss',si+startstep)
