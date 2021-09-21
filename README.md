fjModule_eventData.py is used to create the data sets in nanoAODTools

Use remake_dist.py to make the training data sets for the models. (This also does some preprocessing, so if using different data sets, at least look over the program.)

IN_v5p1 is a pure interaction network that utilizes event data as well as particle data and secondary vertex data.

IN_v4p1 is a pure interaction network. It works extremely well on data sets where there are many particles need for discrimination (hadhad decays for HTauTau for instance).

GRU_v6p1 is a half GRU, half interaction network. It combines PF and SV data similar to the IN network and puts both this and pure SV data into two seperate GRUs which it then combines. It works well on data sets where that are some particularly important particles needed for discrimination (hadel and hadmy decays for HTauTau).
