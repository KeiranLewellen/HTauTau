import ROOT
import numpy as np

ROOT.PyConfig.IgnoreCommandLineOptions = True
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

SVDRMATCH = 0.8
TauDRMATCH = 0.8
GENDRMATCH = 0.8


class pfcandProducer(Module):
    def __init__(self, leptonSel, leptonId=''):
        self.lepSel = leptonSel
        self.leptonId = leptonId
        self.Nparts = 30
        self.Nsvs = 5
        self.Ntaus = 3

    def beginJob(self):
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree

        self.out.branch("fj_idx", "I", 1)
        self.out.branch("fj_msd", "F", 1)
        self.out.branch("fj_LSmsd", "F", 1)
        self.out.branch("fj_m", "F", 1)
        self.out.branch("fj_lsf3", "F", 1)
        self.out.branch("fj_dRLep", "F", 1)
        self.out.branch("fj_pt", "F", 1)
        self.out.branch("fj_LSpt", "F", 1)
        self.out.branch("fj_eta", "F", 1)
        self.out.branch("fj_phi", "F", 1)
        self.out.branch("fj_n2b1", "F", 1)
        self.out.branch("fj_LSn2b1", "F", 1)
        self.out.branch("fj_n3b1", "F", 1)
        self.out.branch("fj_LSn3b1", "F", 1)
        self.out.branch("fj_tau21", "F", 1)
        self.out.branch("fj_tau32", "F", 1)
        self.out.branch("fj_tau43", "F", 1)
        self.out.branch("fj_LStau21", "F", 1)
        self.out.branch("fj_LStau32", "F", 1)
        self.out.branch("fj_LStau43", "F", 1)
        self.out.branch("fj_deepTagZqq", "F", 1)
        self.out.branch("fj_deepTagWqq", "F", 1)
        self.out.branch("fj_had1Decay", "I", 1)
        self.out.branch("fj_had2Decay", "I", 1)
        self.out.branch("fj_nProngs", "I", 1)
        self.out.branch("fj_nEles", "I", 1)
        self.out.branch("fj_nMus", "I", 1)
        self.out.branch("fj_genMass", "F", 1)
        self.out.branch("fj_genPt", "F", 1)
        self.out.branch("fj_genPhi", "F", 1)
        self.out.branch("fj_genEta", "F", 1)
        self.out.branch("fj_genMass", "F", 1)
        self.out.branch("fj_MuonEnergyFraction", "F", 1)
        self.out.branch("fj_ElectronEnergyFraction", "F", 1)
        self.out.branch("fj_PhotonEnergyFraction", "F", 1)
        self.out.branch("fj_ChargedHadronEnergyFraction", "F", 1)
        self.out.branch("fj_NeutralHadronEnergyFraction", "F", 1)
        self.out.branch("fj_MuonNum", "F", 1)
        self.out.branch("fj_ElectronNum", "F", 1)
        self.out.branch("fj_PhotonNum", "F", 1)
        self.out.branch("fj_ChargedHadronNum", "F", 1)
        self.out.branch("fj_NeutralHadronNum", "F", 1)
        self.out.branch("PF_pt", "F", self.Nparts)
        self.out.branch("PF_eta", "F", self.Nparts)
        self.out.branch("PF_phi", "F", self.Nparts)
        self.out.branch("PF_pup", "F", self.Nparts)
        self.out.branch("PF_pupnolep", "F", self.Nparts)
        self.out.branch("PF_q", "F", self.Nparts)
        self.out.branch("PF_id", "F", self.Nparts)
        self.out.branch("PF_trk", "F", self.Nparts)
        self.out.branch("PF_dz", "F", self.Nparts)
        self.out.branch("PF_dxy", "F", self.Nparts)
        self.out.branch("PF_dxyerr", "F", self.Nparts)
        self.out.branch("PF_vtx", "F", self.Nparts)
        self.out.branch("sv_dlen", "F", self.Nsvs)
        self.out.branch("sv_dlenSig", "F", self.Nsvs)
        self.out.branch("sv_dxy", "F", self.Nsvs)
        self.out.branch("sv_dxySig", "F", self.Nsvs)
        self.out.branch("sv_chi2", "F", self.Nsvs)
        self.out.branch("sv_pAngle", "F", self.Nsvs)
        self.out.branch("sv_x", "F", self.Nsvs)
        self.out.branch("sv_y", "F", self.Nsvs)
        self.out.branch("sv_z", "F", self.Nsvs)
        self.out.branch("sv_pt", "F", self.Nsvs)
        self.out.branch("sv_mass", "F", self.Nsvs)
        self.out.branch("sv_eta", "F", self.Nsvs)
        self.out.branch("sv_phi", "F", self.Nsvs)
        self.out.branch("tau_charge", "F", self.Ntaus)
        self.out.branch("tau_chargedIso", "F", self.Ntaus)
        self.out.branch("tau_dxy", "F", self.Ntaus)
        self.out.branch("tau_dz", "F", self.Ntaus)
        self.out.branch("tau_eta", "F", self.Ntaus)
        self.out.branch("tau_leadTkDeltaEta", "F", self.Ntaus)
        self.out.branch("tau_leadTkDeltaPhi", "F", self.Ntaus)
        self.out.branch("tau_leadTkPtOverTauPt", "F", self.Ntaus)
        self.out.branch("tau_mass", "F", self.Ntaus)
        self.out.branch("tau_neutralIso", "F", self.Ntaus)
        self.out.branch("tau_phi", "F", self.Ntaus)
        self.out.branch("tau_photonsOutsideSignalCone", "F", self.Ntaus)
        self.out.branch("tau_pt", "F", self.Ntaus)
        self.out.branch("tau_rawAntiEle", "F", self.Ntaus)
        self.out.branch("tau_rawIso", "F", self.Ntaus)
        self.out.branch("tau_rawIsodR03", "F", self.Ntaus)
        self.out.branch("tau_rawMVAoldDM2017v2", "F", self.Ntaus)
        self.out.branch("tau_rawMVAoldDMdR032017v2", "F", self.Ntaus)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def searchForTauMom(self, thispart, partlist, stopids):
        if thispart.genPartIdxMother in stopids:
            return thispart.genPartIdxMother
        elif thispart.genPartIdxMother >= 0:
            return self.searchForTauMom(partlist[thispart.genPartIdxMother], partlist, stopids)
        else:
            return -1

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        pfcands = Collection(event, "FatJetPFCands")
        jets = Collection(event, "FatJet")
        try:
            genparts = Collection(event, "GenPart")
        except:
            genparts = []
        SV = Collection(event, "SV")
        Tau = Collection(event, "Tau")
        try:
            genVisTaus = Collection(event, "GenVisTau")
        except:
            genVisTaus = []
        muons = Collection(event, "Muon")
        elecs = Collection(event, "Electron")
        met = Object(event, "MET")

        nGoodMus = 0
        nGoodEls = 0
        # vetoes jets without a muon
        if self.leptonId == 'mu':
            for mu in muons:
                if (mu.looseId and mu.pt > 30. and abs(mu.eta) < 2.4):
                    nGoodMus = nGoodMus + 1
            if nGoodMus == 0:
                return False
        # vetoes jets without an electron
        if self.leptonId == 'el':
            for el in elecs:
                if (el.mvaFall17V2noIso_WP90 and el.pt > 40. and abs(el.eta) < 2.5):
                    nGoodEls = nGoodEls + 1
            if nGoodEls == 0:
                return False
        # vetoes jets with leptons
        if self.leptonId == "noLep":
            for mu in muons:
                if (mu.looseId and mu.pt > 10. and abs(mu.eta) < 2.4):
                    nGoodMus = nGoodMus + 1
            if nGoodMus > 0:
                return False
            for el in elecs:
                if (el.mvaFall17V2noIso_WP90 and el.pt > 10. and abs(el.eta) < 2.5):
                    nGoodEls = nGoodEls + 1
            if nGoodEls > 0:
                return False
        # vetoes jets without exactly one muon
        if self.leptonId == 'oneMu':
            for mu in muons:
                if (mu.looseId and mu.pt > 30. and abs(mu.eta) < 2.4):
                    nGoodMus = nGoodMus + 1
            if nGoodMus != 1:
                return False
        # vetoes jets without exactly one electron
        if self.leptonId == 'oneEl':
            for el in elecs:
                if (el.mvaFall17V2noIso_WP90 and el.pt > 40. and abs(el.eta) < 2.5):
                    nGoodEls = nGoodEls + 1
            if nGoodEls != 1:
                return False

        # actually use met arbitration
        jet_idx = -1
        min_dphi = 999.
        for ij, jet in enumerate(jets):
            if (jet.pt < 200.): continue
            this_dphi = abs(signedDeltaPhi(met.phi, jet.phi))
            if (this_dphi < min_dphi):
                min_dphi = this_dphi
                jet_idx = ij

        candit = 0
        for ij, jet in enumerate(jets):

            # if jet.pt < 400 or jet.msoftdrop < 30 : continue
            if (ij < jet_idx):
                candit = candit + jet.nPFConstituents
                continue
            elif (ij > jet_idx):
                continue
            if jet.nPFConstituents < 1: continue

            ##Fill basic jet properties
            jpt = jet.pt
            jLSpt = jet.LSpt
            jeta = jet.eta
            jphi = jet.phi
            jmsd = jet.msoftdrop
            jLSmsd = jet.LSmsoftdrop
            jm = jet.mass
            jdRLep = jet.dRLep
            jlsf3 = jet.lsf3
            jn2b1 = jet.n2b1
            jLSn2b1 = jet.LSn2b1
            jdeepTagZqq = jet.deepTagZqq
            jdeepTagWqq = jet.deepTagWqq
            jn3b1 = jet.n3b1
            jLSn3b1 = jet.LSn3b1


            try:
                jtau21 = float(jet.tau2) / float(jet.tau1)
            except:
                jtau21 = 0.
            try:
                jtau32 = float(jet.tau3) / float(jet.tau2)
            except:
                jtau32 = 0.
            try:
                jtau43 = float(jet.tau4) / float(jet.tau3)
            except:
                jtau43 = 0.
            try:
                jLStau21 = float(jet.LStau2) / float(jet.LStau1)
            except:
                jLStau21 = 0.
            try:
                jLStau32 = float(jet.LStau3) / float(jet.LStau2)
            except:
                jLStau32 = 0.
            try:
                jLStau43 = float(jet.LStau4) / float(jet.LStau3)
            except:
                jLStau43 = 0.

            ##Calculate # prongs by looping over daughters of Z'
            status = 0
            motheridx = None
            jetv = ROOT.TLorentzVector()
            jetv.SetPtEtaPhiM(jpt, jeta, jphi, jmsd)
            dau1 = ROOT.TLorentzVector()
            dau2 = ROOT.TLorentzVector()
            hpt = 0.
            heta = 0.
            hphi = 0.
            gMass = 0.

            for ig, gpart in enumerate(genparts):
                # if (abs(gpart.pdgId)==13 and gpart.genPartIdxMother>=0):
                #    print('momid',genparts[gpart.genPartIdxMother].pdgId)
                #    if ((genparts[gpart.genPartIdxMother].pdgId)==15 and genparts[gpart.genPartIdxMother].genPartIdxMother>=0): print('gmomid',genparts[genparts[gpart.genPartIdxMother].genPartIdxMother].pdgId)
                if (gpart.pdgId == 25 or gpart.pdgId == 23 or gpart.pdgId == 5000001) and gpart.status > status:
                    gMass = gpart.mass
                    motheridx = ig
                    hpt = gpart.pt
                    hphi = gpart.phi
                    heta = gpart.eta
                    status = gpart.status

            # print('status',status,'motheridx',motheridx)

            dauidx1 = None
            dauidx2 = None

            for ig, gpart in enumerate(genparts):
                if motheridx is None: break
                # if gpart.genPartIdxMother == motheridx and abs(gpart.pdgId)==15:
                if gpart.genPartIdxMother == motheridx and abs(gpart.pdgId) == self.lepSel:
                    if dau1.Pt() == 0.:
                        dau1.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass)
                        dauidx1 = ig
                    elif dau2.Pt() == 0.:
                        dau2.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass)
                        dauidx2 = ig
                if dauidx1 is not None:
                    if gpart.genPartIdxMother == dauidx1 and gpart.pdgId == genparts[dauidx1].pdgId:
                        dau1.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass)
                        dauidx1 = ig
                if dauidx2 is not None:
                    if gpart.genPartIdxMother == dauidx2 and gpart.pdgId == genparts[dauidx2].pdgId:
                        dau2.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass)
                        dauidx2 = ig

            # print('dauidx1',dauidx1,'dauidx2',dauidx2)

            nEle = 0
            nMu = 0

            if dauidx1 is not None and dauidx2 is not None:
                for ig, gpart in enumerate(genparts):
                    matchidx = self.searchForTauMom(gpart, genparts, [dauidx1, dauidx2])
                    # if (matchidx>-1):
                    #    print('matchidx',matchidx,'pdgid',gpart.pdgId)
                    if matchidx == dauidx1 and abs(gpart.pdgId) in [12, 14, 16]:
                        neutrino = ROOT.TLorentzVector()
                        neutrino.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass)
                        dau1 = dau1 - neutrino
                        if abs(gpart.pdgId) == 12:
                            nEle = nEle + 1
                        elif abs(gpart.pdgId) == 14:
                            nMu = nMu + 1
                    if matchidx == dauidx2 and abs(gpart.pdgId) in [12, 14, 16]:
                        neutrino = ROOT.TLorentzVector()
                        neutrino.SetPtEtaPhiM(gpart.pt, gpart.eta, gpart.phi, gpart.mass)
                        dau2 = dau2 - neutrino
                        if abs(gpart.pdgId) == 12:
                            nEle = nEle + 1
                        elif abs(gpart.pdgId) == 14:
                            nMu = nMu + 1

            nProngs = 0
            if dau1.Pt() > 0. and dau2.Pt() > 0.:
                if jetv.DeltaR(dau1) < GENDRMATCH: nProngs += 1
                if jetv.DeltaR(dau2) < GENDRMATCH: nProngs += 1

            # print(nProngs,nEle,nMu,(dauidx1 is not None and dauidx2 is not None))

            hadDecay = [-1, -1]

            for ig, gt in enumerate(genVisTaus):
                if (ig >= nProngs): break
                tauv = ROOT.TLorentzVector()
                tauv.SetPtEtaPhiM(gt.pt, gt.eta, gt.phi, gt.mass)
                if jetv.DeltaR(tauv) < GENDRMATCH:
                    hadDecay[ig] = gt.status
            # 0=OneProng0PiZero, 1=OneProng1PiZero, 2=OneProng2PiZero, 10=ThreeProng0PiZero, 11=ThreeProng1PiZero, 15=Other

            ##Fill SV
            svpt = np.zeros(self.Nsvs, dtype=np.float16)
            svdlen = np.zeros(self.Nsvs, dtype=np.float16)
            svdlenSig = np.zeros(self.Nsvs, dtype=np.float16)
            svdxy = np.zeros(self.Nsvs, dtype=np.float16)
            svdxySig = np.zeros(self.Nsvs, dtype=np.float16)
            svchi2 = np.zeros(self.Nsvs, dtype=np.float16)
            svpAngle = np.zeros(self.Nsvs, dtype=np.float16)
            svx = np.zeros(self.Nsvs, dtype=np.float16)
            svy = np.zeros(self.Nsvs, dtype=np.float16)
            svz = np.zeros(self.Nsvs, dtype=np.float16)
            svmass = np.zeros(self.Nsvs, dtype=np.float16)
            svphi = np.zeros(self.Nsvs, dtype=np.float16)
            sveta = np.zeros(self.Nsvs, dtype=np.float16)
            svv = ROOT.TLorentzVector()
            arrIdx = 0
            for isv, sv in enumerate(SV):
                if arrIdx == self.Nsvs: break
                svv.SetPtEtaPhiM(sv.pt, sv.eta, sv.phi, sv.mass)
                if jetv.DeltaR(svv) < SVDRMATCH:
                    svpt[arrIdx] = sv.pt / jpt
                    svdlen[arrIdx] = sv.dlen
                    svdlenSig[arrIdx] = sv.dlenSig
                    svdxy[arrIdx] = sv.dxy
                    svdxySig[arrIdx] = sv.dxySig
                    svchi2[arrIdx] = sv.chi2
                    svpAngle[arrIdx] = sv.pAngle
                    svx[arrIdx] = sv.x
                    svy[arrIdx] = sv.y
                    svz[arrIdx] = sv.z
                    sveta[arrIdx] = sv.eta - jeta
                    svphi[arrIdx] = signedDeltaPhi(sv.phi, jphi)
                    svmass[arrIdx] = sv.mass
                    arrIdx += 1

            # Fill Taus
            Tau_charge = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_chargedIso = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_dxy = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_dz = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_eta = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_leadTkDeltaEta = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_leadTkDeltaPhi = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_leadTkPtOverTauPt = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_mass = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_neutralIso = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_phi = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_photonsOutsideSignalCone = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_pt = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_rawAntiEle = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_rawIso = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_rawIsodR03 = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_rawMVAoldDM2017v2 = np.zeros(self.Ntaus, dtype=np.float16)
            Tau_rawMVAoldDMdR032017v2 = np.zeros(self.Ntaus, dtype=np.float16)
            tauv = ROOT.TLorentzVector()
            tauIdx = 0
            for tau in Tau:
                if tauIdx == self.Ntaus:
                    break
                tauv.SetPtEtaPhiM(tau.pt, tau.eta, tau.phi, tau.mass)
                if jetv.DeltaR(tauv) < TauDRMATCH:
                    Tau_charge[tauIdx] = tau.charge
                    Tau_chargedIso[tauIdx] = tau.chargedIso / tau.pt
                    Tau_dxy[tauIdx] = tau.dxy
                    Tau_dz[tauIdx] = tau.dz
                    Tau_eta[tauIdx] = tau.eta - jeta
                    Tau_leadTkDeltaEta[tauIdx] = tau.leadTkDeltaEta
                    Tau_leadTkDeltaPhi[tauIdx] = tau.leadTkDeltaPhi
                    Tau_leadTkPtOverTauPt[tauIdx] = tau.leadTkPtOverTauPt
                    Tau_mass[tauIdx] = tau.mass
                    Tau_neutralIso[tauIdx] = tau.neutralIso / tau.pt
                    Tau_phi[tauIdx] = signedDeltaPhi(tau.phi, jphi)
                    Tau_photonsOutsideSignalCone[tauIdx] = tau.photonsOutsideSignalCone
                    Tau_pt[tauIdx] = tau.pt / jpt
                    Tau_rawAntiEle[tauIdx] = tau.rawAntiEle
                    Tau_rawIso[tauIdx] = tau.rawIso / tau.pt
                    Tau_rawIsodR03[tauIdx] = tau.rawIsodR03
                    Tau_rawMVAoldDM2017v2[tauIdx] = tau.rawMVAoldDM2017v2
                    Tau_rawMVAoldDMdR032017v2[tauIdx] = tau.rawMVAoldDMdR032017v2
                    tauIdx += 1

            ##Fill PF candidates
            candrange = range(candit, candit + jet.nPFConstituents)
            pfpt = np.zeros(self.Nparts, dtype=np.float16)
            pfeta = np.zeros(self.Nparts, dtype=np.float16)
            pfphi = np.zeros(self.Nparts, dtype=np.float16)
            pftrk = np.zeros(self.Nparts, dtype=np.float16)
            pfpup = np.zeros(self.Nparts, dtype=np.float16)
            pfpupnolep = np.zeros(self.Nparts, dtype=np.float16)
            pfq = np.zeros(self.Nparts, dtype=np.float16)
            pfid = np.zeros(self.Nparts, dtype=np.float16)
            pfdz = np.zeros(self.Nparts, dtype=np.float16)
            pfdxy = np.zeros(self.Nparts, dtype=np.float16)
            pfdxyerr = np.zeros(self.Nparts, dtype=np.float16)
            pfvtx = np.zeros(self.Nparts, dtype=np.float16)
            arrIdx = 0
            jMuonEnergy = np.float16(0)
            jElectronEnergy = np.float16(0)
            jPhotonEnergy = np.float16(0)
            jChargedHadronEnergy = np.float16(0)
            jNeutralHadronEnergy = np.float16(0)
            jMuonNum = np.float16(0)
            jElectronNum = np.float16(0)
            jPhotonNum = np.float16(0)
            jChargedHadronNum = np.float16(0)
            jNeutralHadronNum = np.float16(0)

            for ip, part in enumerate(pfcands):
                if ip not in candrange: continue
                if arrIdx < self.Nparts:
                    pfpt[arrIdx] = part.pt / jpt
                    pfeta[arrIdx] = part.eta - jeta
                    pfphi[arrIdx] = signedDeltaPhi(part.phi, jphi)
                    pfpup[arrIdx] = part.puppiWeight
                    pfpupnolep[arrIdx] = part.puppiWeightNoLep
                    pfq[arrIdx] = part.charge
                    pfid[arrIdx] = part.pdgId
                    pfdz[arrIdx] = part.dz
                    pfdxy[arrIdx] = part.d0
                    pfdxyerr[arrIdx] = part.d0Err
                    pftrk[arrIdx] = part.trkChi2
                    pfvtx[arrIdx] = part.vtxChi2
                if part.pdgId in [-13., 13.]:
                    jMuonEnergy += part.pt
                    jMuonNum += 1
                if part.pdgId in [-11., 11.]:
                    jElectronEnergy += part.pt
                    jElectronNum += 1
                if part.pdgId in [22.]:
                    jPhotonEnergy += part.pt
                    jPhotonNum += 1
                if part.pdgId in [-211., 211.]:
                    jChargedHadronEnergy += part.pt
                    jChargedHadronNum += 1
                if part.pdgId in [-111., 111.,  130.]:
                    jNeutralHadronEnergy += part.pt
                    jNeutralHadronNum += 1
                arrIdx += 1

            if jmsd == 0:
                jLSmsd = np.inf
            else:
                jLSmsd = jLSmsd/jmsd

            self.out.fillBranch("fj_idx", ij)
            self.out.fillBranch("fj_pt", jpt)
            self.out.fillBranch("fj_LSpt", jLSpt/jpt)
            self.out.fillBranch("fj_eta", jeta)
            self.out.fillBranch("fj_phi", jphi)
            self.out.fillBranch("fj_lsf3", jlsf3)
            self.out.fillBranch("fj_dRLep", jdRLep)
            self.out.fillBranch("fj_n2b1", jLSn2b1)
            self.out.fillBranch("fj_LSn2b1", jLSn2b1)
            self.out.fillBranch("fj_n3b1", jn3b1)
            self.out.fillBranch("fj_LSn3b1", jLSn3b1)
            self.out.fillBranch("fj_tau21", jtau21)
            self.out.fillBranch("fj_tau32", jtau32)
            self.out.fillBranch("fj_tau43", jtau43)
            self.out.fillBranch("fj_LStau21", jLStau21)
            self.out.fillBranch("fj_LStau32", jLStau32)
            self.out.fillBranch("fj_LStau43", jLStau43)
            self.out.fillBranch("fj_had1Decay", hadDecay[0])
            self.out.fillBranch("fj_had2Decay", hadDecay[1])
            self.out.fillBranch("fj_nProngs", nProngs)
            self.out.fillBranch("fj_nEles", nEle)
            self.out.fillBranch("fj_nMus", nMu)
            self.out.fillBranch("fj_genMass", gMass)
            self.out.fillBranch("fj_genPt", hpt)
            self.out.fillBranch("fj_genEta", heta)
            self.out.fillBranch("fj_genPhi", hphi)
            self.out.fillBranch("fj_deepTagZqq", jdeepTagZqq)
            self.out.fillBranch("fj_deepTagWqq", jdeepTagWqq)
            self.out.fillBranch("fj_msd", jmsd)
            self.out.fillBranch("fj_LSmsd", jLSmsd)
            self.out.fillBranch("fj_m", jm)
            self.out.fillBranch("fj_MuonEnergyFraction", jMuonEnergy/jpt)
            self.out.fillBranch("fj_ElectronEnergyFraction", jElectronEnergy/jpt)
            self.out.fillBranch("fj_PhotonEnergyFraction", jPhotonEnergy/jpt)
            self.out.fillBranch("fj_ChargedHadronEnergyFraction", jChargedHadronEnergy/jpt)
            self.out.fillBranch("fj_NeutralHadronEnergyFraction", jNeutralHadronEnergy/jpt)
            self.out.fillBranch("fj_MuonNum", jMuonNum)
            self.out.fillBranch("fj_ElectronNum", jElectronNum)
            self.out.fillBranch("fj_PhotonNum", jPhotonNum)
            self.out.fillBranch("fj_ChargedHadronNum", jChargedHadronNum)
            self.out.fillBranch("fj_NeutralHadronNum", jNeutralHadronNum)
            self.out.fillBranch("sv_dlen", svdlen)
            self.out.fillBranch("sv_dlenSig", svdlenSig)
            self.out.fillBranch("sv_dxy", svdxy)
            self.out.fillBranch("sv_dxySig", svdxySig)
            self.out.fillBranch("sv_chi2", svchi2)
            self.out.fillBranch("sv_pAngle", svpAngle)
            self.out.fillBranch("sv_x", svx)
            self.out.fillBranch("sv_y", svy)
            self.out.fillBranch("sv_z", svz)
            self.out.fillBranch("sv_pt", svpt)
            self.out.fillBranch("sv_mass", svmass)
            self.out.fillBranch("sv_eta", sveta)
            self.out.fillBranch("sv_phi", svphi)
            self.out.fillBranch("PF_pt", pfpt)
            self.out.fillBranch("PF_eta", pfeta)
            self.out.fillBranch("PF_phi", pfphi)
            self.out.fillBranch("PF_pup", pfpup)
            self.out.fillBranch("PF_pupnolep", pfpupnolep)
            self.out.fillBranch("PF_q", pfq)
            self.out.fillBranch("PF_id", pfid)
            self.out.fillBranch("PF_dz", pfdz)
            self.out.fillBranch("PF_dxy", pfdxy)
            self.out.fillBranch("PF_dxyerr", pfdxyerr)
            self.out.fillBranch("PF_trk", pftrk)
            self.out.fillBranch("PF_vtx", pfvtx)
            self.out.fillBranch("tau_charge", Tau_charge)
            self.out.fillBranch("tau_chargedIso", Tau_chargedIso)
            self.out.fillBranch("tau_dxy", Tau_dxy)
            self.out.fillBranch("tau_dz", Tau_dz)
            self.out.fillBranch("tau_eta", Tau_eta)
            self.out.fillBranch("tau_leadTkDeltaEta", Tau_leadTkDeltaEta)
            self.out.fillBranch("tau_leadTkDeltaPhi", Tau_leadTkDeltaPhi)
            self.out.fillBranch("tau_leadTkPtOverTauPt", Tau_leadTkPtOverTauPt)
            self.out.fillBranch("tau_mass", Tau_mass)
            self.out.fillBranch("tau_neutralIso", Tau_neutralIso)
            self.out.fillBranch("tau_phi", Tau_phi)
            self.out.fillBranch("tau_photonsOutsideSignalCone", Tau_photonsOutsideSignalCone)
            self.out.fillBranch("tau_pt", Tau_pt)
            self.out.fillBranch("tau_rawAntiEle", Tau_rawAntiEle)
            self.out.fillBranch("tau_rawIso", Tau_rawIso)
            self.out.fillBranch("tau_rawIsodR03", Tau_rawIsodR03)
            self.out.fillBranch("tau_rawMVAoldDM2017v2", Tau_rawMVAoldDM2017v2)
            self.out.fillBranch("tau_rawMVAoldDMdR032017v2", Tau_rawMVAoldDMdR032017v2)
            return True
        return False


def signedDeltaPhi(phi1, phi2):
    dPhi = phi1 - phi2
    if (dPhi < -np.pi):
        dPhi = 2 * np.pi + dPhi
    elif (dPhi > np.pi):
        dPhi = -2 * np.pi + dPhi
    return dPhi


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
pfModule = lambda: pfcandProducer(leptonSel=15)
pfModule_el = lambda: pfcandProducer(leptonSel=15, leptonId='el')
pfModule_mu = lambda: pfcandProducer(leptonSel=15, leptonId='mu')
pfModule_noLep = lambda: pfcandProducer(leptonSel=15, leptonId='noLep')
pfModule_oneEl = lambda: pfcandProducer(leptonSel=11, leptonId='oneEl')
pfModule_oneMu = lambda: pfcandProducer(leptonSel=13, leptonId='oneMu')
