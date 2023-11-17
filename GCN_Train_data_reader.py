import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import csv
import re
import logging
from dataclasses import dataclass

import deepnovo_config
from deepnovo_cython_modules import get_ion_index, process_peaks

logger = logging.getLogger(__name__)

@dataclass
class DDAFeature:
    feature_id: str
    mz: float
    z: float
    rt_mean: float
    peptide: list
    scan: str
    mass: float
    feature_area: str

@dataclass
class TrainData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    precursormass: float
    spectrum_representation: np.ndarray
    forward_id_target: list
    backward_id_target: list
    forward_ion_location_index_list: list
    backward_ion_location_index_list: list
    forward_id_input: list
    backward_id_input: list

@dataclass
class Tag:
    tag : list
    score : list
    moz: np.ndarray
    intensity : np.ndarray
    label:int

@dataclass
class TagTest:
    tag : list
    score : list
    moz: np.ndarray
    intensity : np.ndarray


def parse_raw_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'N(Deamidation)'
                index += 6
            elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'Q(Deamidation)'
                index += 6
            else:  # unknown modification
                logger.warning(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1

    return True, peptide
# print(parse_raw_sequence('AIIISC(+57.02)TYIK'))


class GCNTagTrainDataset(Dataset):
    def __init__(self, feature_filename, spectrum_filename, args, tagpath, transform=None):
        print("run init ")
        """
        read all feature information and store in memory,
        :param feature_filename:
        :param spectrum_filename:
        """
        print('start')
        logger.info(f"input spectrum file: {spectrum_filename}")
        logger.info(f"input feature file: {feature_filename}")
        self.args = args
        self.spectrum_filename = spectrum_filename
        self.input_spectrum_handle = None
        self.feature_list = []
        self.spectrum_location_dict = {}
        self.transform = transform
        self.sequence_dict, self.moz_dict, self.instensity_dict = self.read_mgf(spectrum_filename)
        self.scan_no_list, self.tag_list, self.tag_score_list = read_tagbeamserch(tagpath,self.sequence_dict)



    def __len__(self):
        # print(len(self.tag_list))
        return len(self.tag_list)

    def close(self):
        self.input_spectrum_handle.close()

    def _parse_spectrum_ion_tag(self):
        mz_list = []
        intensity_list = []
        line = self.input_spectrum_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX

            if mz_float > self.args.MZ_MAX:
                line = self.input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(intensity_float)
            # intensity_list.append(intensity_float)
            # intensity_list.append(intensity_float)
            line = self.input_spectrum_handle.readline()
        return mz_list, intensity_list


    def _parse_spectrum_ion(self):
        mz_list = []
        intensity_list = []
        line = self.input_spectrum_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX

            if mz_float > self.args.MZ_MAX:
                line = self.input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(math.sqrt(intensity_float))
            # intensity_list.append(intensity_float)
            # intensity_list.append(intensity_float)
            line = self.input_spectrum_handle.readline()
        return mz_list, intensity_list


    def read_mgf(self,path):
        sequence_dict = {}
        moz_dict = {}
        instensity_dict = {}

        moz_ls = []
        intensity_ls = []
        with open(path, "r") as f:
            content = f.readlines()
            for line in content:
                if line.startswith("P"):
                    pepmass = float(line[:-1].split("=")[-1])
                elif line.startswith("SCAN"):
                    scan = line[:-1].split("=")[-1]
                    moz_ls = []
                    intensity_ls = []

                elif line.startswith("SEQ"):
                    seq = line[:-1].split("=")[-1]
                    new_seq = GetIncludeModSequcen(seq)
                    sequence_dict[scan] = new_seq
                    # print(seq,new_seq)
                elif line.startswith("BEG") or line.startswith("T") or line.startswith(
                        "C") or line.startswith("R"):
                    continue
                elif line.startswith("END"):
                    peak_location, peak_intensity, spectrum_representation = process_peaks(moz_ls, intensity_ls, pepmass, self.args)
                    moz_dict[scan] = peak_location
                    instensity_dict[scan] = peak_intensity
                else:
                    line_ls = line.split(" ")
                    # print(line_ls)
                    moz = float(line_ls[0])
                    intensity = float(line_ls[1][:-1])
                    # print(moz,intensity)
                    moz_ls.append(moz)
                    intensity_ls.append(math.sqrt(intensity))

        # print("sequence_dict",type(sequence_dict),sequence_dict)
        # print("moz_dict",moz_dict)
        return sequence_dict, moz_dict,instensity_dict


    def __getlabel_(self, tag, sequnce):
        flag = 0
        sequnce = sequnce.replace("L", "I")
        tag = tag.replace("L", "I")
        tag_re = tag[::-1]
        if tag_re in sequnce or tag_re in sequnce:
            flag = 1
        return flag

    def _get_TagFeature(self,idx):
        tag = self.tag_list[idx]
        ls1 = [0 for x in range(40)]
        ls2 = [0 for x in range(40)]
        for aidx,aa in enumerate(tag):
            if aa == "C":
                aa = "C(Carbamidomethylation)"
            aa_id = deepnovo_config.vocab_reverse.index(aa)
            # print(aidx,tag)
            ls1[aidx] = aa_id
        score_list = self.tag_score_list[idx]
        for s_idx,score in enumerate(score_list):
            ls2[s_idx] = score

        scan_no = self.scan_no_list[idx]
        sequence = self.sequence_dict[scan_no]
        moz = self.moz_dict[scan_no]
        intensity = self.instensity_dict[scan_no]
        label = self.__getlabel_(tag,sequence)
        return Tag(ls1,
                   ls2,
                   moz,
                   intensity,
                   label)

    def __getitem__(self, idx):

        return self._get_TagFeature(idx)


def collate_func(train_data_list):
    """
    :param train_data_list: list of TrainData
    :return:
        peak_location: [batch, N]
        peak_intensity: [batch, N]
        forward_target_id: [batch, T]
        backward_target_id: [batch, T]
        forward_ion_index_list: [batch, T, 26, 8]
        backward_ion_index_list: [batch, T, 26, 8]
    """
    # sort data by seq length (decreasing order)
    # print("start collate_func")
    # print("traindatalist",train_data_list)
    i = 0
    x_data = []
    y_data = []
    for tagclass in train_data_list:
        tag = tagclass.tag
        tag_np = np.array(tag)
        score = tagclass.score
        score_np = np.array(score)
        moz = tagclass.moz
        intensity = tagclass.intensity
        # print("tag",tag)
        # print("score",score)
        # print("moz",moz)
        # print(intensity)
        # print("tag_np:", tag_np.shape)
        # print("score_np", score_np.shape)

        x = np.append(tag_np,score_np)
        # print("x:",x.shape)
        x = np.append(x,moz)
        # print("x:", x.shape)
        x = np.append(x, intensity)
        # print("x:", x.shape)
        x_data.append(x)
        la = tagclass.label
        y = np.array(la)
        y_data.append([y])
    x_data = np.stack(x_data)
    x_data = torch.from_numpy(x_data)

    y_data = np.stack(y_data)
    y_data = torch.from_numpy(y_data)
    # print("x_data,", x_data.shape,x_data)
    # print("y_data,", y_data.shape,y_data)
    return (x_data, y_data)


def ReadPKL(path):
    """
    r_path1 = path+"/dict_titleToscan.pkl"
    f1 = open(r_path1, 'rb')
    dict_titleToscan = pickle.load(f1)
    # print(dict_titleToscan)
    """

    r_path2 = path + "/dict_scantosequece.pkl"
    f2 = open(r_path2, 'rb')
    dict_scantosequece = pickle.load(f2)
    # print(dict_scantosequece)

    """
    r_path3 = path + "/dict_titletosequece.pkl"
    f3 = open(r_path3, 'rb')
    dict_titletosequece = pickle.load(f3)
    # print(dict_titletosequece)
    """
    return dict_scantosequece


def remove_bracket_content(s):
    return re.sub(r'\(.*?\)', '', s)


def read_tagbeamserch(path, seuqece_dict):
    tag_list = []
    tag_score_list=[]
    scan_no_list=[]
    cont_correct = 0
    cont_error = 0
    with open(path, "r") as f:
        content = f.readlines()
        for idx, line in enumerate(content):
            if line.startswith("BEGIN"):
                scan_no = content[idx+1].split("\t")[0]
                scan_no_idx = idx+1
                cont = 0

            elif line[0].isdigit() and idx != scan_no_idx:

                line_ls = line.split("\t")
                tag_sequece = line_ls[2]
                # print("tagseunce:",tag_sequece)
                tag_sequece = remove_bracket_content(tag_sequece)
                # print("tagseunce:", tag_sequece)
                if len(tag_sequece)<3:
                    continue
                tag_sequece = tag_sequece.replace(",", "")
                # print(tag_sequece)
                tag_re = tag_sequece[::-1]

                sequcee = seuqece_dict[scan_no]
                if tag_re in tag_sequece or tag_re in sequcee:
                    cont_correct += 1

                    tag_aa_score_ls = [float(x) for x in line_ls[4].split(",")]

                    scan_no_list.append(scan_no)
                    tag_list.append(tag_sequece)
                    tag_score_list.append(tag_aa_score_ls)
                else:
                    if cont_error > cont_correct/2:
                        continue
                    else:
                        tag_aa_score_ls = [float(x) for x in line_ls[4].split(",")]

                        scan_no_list.append(scan_no)
                        tag_list.append(tag_sequece)
                        tag_score_list.append(tag_aa_score_ls)
                        cont_error += 1

    # print("len",len(scan_no_list),len(tag_list),len(tag_score_list))
    print("cpn",cont_error,cont_correct)

    # print("scan_no_list",scan_no_list) #['F26:16935', 'F26:16935', 'F26:16935', 'F26:16935', 'F26:16935', 'F26:16935', 'F26:16935', 'F26:16935', 'F26:16935', 'F26:16935', 'F26:16935', 'F26:16935', 'F26:16935']
    # print("tag_list",tag_list) # [..'IFI', 'IHFIQTIA', 'IAG', 'TIA', 'IHFRQ', 'GIMDSFVNDIF', 'DSFVNDIF', 'SFVNDIF', 'HFSDMIG', 'DIF', 'HFSD', 'HFS', 'GIMDSFVNDIFER', 'DSFVNDIFER', 'DIFER', 'FNG', 'MVSR', 'VSR', 'SSR', 'GMMPSFVNDIFER', 'NDSFVNDIFER']
    # print("tag_score_list", tag_score_list) #[.. [-0.33661073446273804, -0.39711251854896545, -2.415278673171997], [-0.47913387417793274, -0.3402237594127655, -3.3808462619781494, -4.768370445162873e-07], ..]
    return scan_no_list, tag_list, tag_score_list


def GetIncludeModSequcen(seq):
    num = seq.count("(")
    # print(num)
    flag = False
    seq_new = ""
    for j,aa in enumerate(seq):
        if aa == "(":
            flag=True
            if seq[j-1] == "N":
                seq_new += "(Deamidated)"
            elif seq[j-1] == "Q":
                seq_new += "(Deamidated)"
            elif seq[j-1] == "M":
                seq_new += "(Oxidation)"
        if flag == False:
            seq_new += aa
        if aa == ")":
            flag = False
    return seq_new


def read_mgf(path):
    sequence_dict = {}
    moz_dict = {}
    instensity_dict = {}

    moz_ls = []
    intensity_ls = []
    with open(path, "r") as f:
        content = f.readlines()
        for line in content:
            if line.startswith("P"):
                pepmass = float(line[:-1].split("=")[-1])
                # print(pepmass)
            elif line.startswith("SCAN"):
                scan = line[:-1].split("=")[-1]
                moz_ls = []
                intensity_ls = []

            elif line.startswith("SEQ"):
                seq = line[:-1].split("=")[-1]
                new_seq = GetIncludeModSequcen(seq)
                sequence_dict[scan] = new_seq
                # print(seq,new_seq)
            elif line.startswith("BEG") or line.startswith("T") or line.startswith("C") or line.startswith("R"):
                continue
            elif line.startswith("END"):
                moz_dict[scan] = moz_ls
                instensity_dict[scan] = intensity_ls
            else:
                line_ls = line.split(" ")
                # print(line_ls)
                moz = float(line_ls[0])
                intensity = float(line_ls[1][:-1])
                # print(moz,intensity)
                moz_ls.append(moz)
                intensity_ls.append(intensity)
    print(sequence_dict)
    return sequence_dict
    # print(moz_dict)

def _get_TagFeature(tag_list,idx):
    tag = tag_list[idx]
    # tag = "ICII"
    ls1 = [0 for x in range(25)]
    for aidx,aa in enumerate(tag):
        if aa == "C":
            aa = "C(Carbamidomethylation)"
        aa_id = deepnovo_config.vocab_reverse.index(aa)
        ls1[aidx] = aa_id
    print(ls1)

if __name__=="__main__":

    # _get_TagFeature(tag_list,125)
    sequence_dict = read_mgf("E:/MyProject/GCNTag_data/PDX004565/cross.9high_80k.exclude_bacillus/spectrum_test.mgf")
    scan_no_list, tag_list, tag_score_list = read_tagbeamserch("./PXD004565/test_20000_Data.tagbeamsearch", sequence_dict)