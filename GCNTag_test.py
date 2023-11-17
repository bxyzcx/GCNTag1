import re

import torch
from torch.utils.data import Dataset
import time
import logging
import logging.config
import deepnovo_config
import os
import datetime
from init_args import init_args
import math
from train_func1 import train, build_model, validation, perplexity
from deepnovo_cython_modules import get_ion_index, process_peaks
logger = logging.getLogger(__name__)
from GCN_Train_data_reader import GCNTagTrainDataset,ReadPKL,collate_func
from torch import nn  # 完成神经欧网络的相关工作
from torch.nn import functional as F  # 常用的函数
from torch import optim  # 工具包
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass




class Net(nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()

        # xw+b
        input_size = args.MAX_NUM_PEAK * 2 + 80
        self.fc1 = nn.Linear(input_size, 256)  #第一层 256 随机指定
        # self.fc1 = nn.Linear(580, 1)  #第一层 256 随机指定
        self.bacth1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bacth2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)  # 第三层 因为是个十个分类所以最后应该是个10

    def forward(self, x):
        # x = F.relu((self.fc1(x)))
        # x = F.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))

        x = F.relu(self.bacth1((self.fc1(x))))
        x = F.relu(self.bacth2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x

@dataclass
class TagResualt:
    direction:str
    tag_seq:str
    tag_score:str
    tag_position_score:list
    left:str
    rigth:str



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

def read_mgf(path,args):
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
                # print(seq,new_seq)
            elif line.startswith("BEG") or line.startswith("T") or line.startswith(
                    "C") or line.startswith("R"):
                continue
            elif line.startswith("END"):
                peak_location, peak_intensity, spectrum_representation = process_peaks(moz_ls, intensity_ls, pepmass,
                                                                                       args)
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
    return  moz_dict, instensity_dict



def get_fixedTag(tag,tag_score,tag_list,tag_score_list,Wsize, left, right,direction,tagResulet_ls):
    """
    tag:APPTID
    tag_score:[-0.146,-4213.,...]
    tag_list:当前谱图已存的全部tag  为了存储tag是判断是否已经存在过
    tag_score_list:当前谱图已存的全部tag对应的打分  为了后续进行tag排序
    Wsize:提取定长标签的长度

    """
    # print("tag",tag)
    for ti,t in enumerate(tag):
        aa_mass =0.0
        new_socre = []
        score = 0.0
        new_tag = ""
        try:
            for s in range(Wsize):
                new_tag += tag[ti+s]
                if tag[ti+s] == "C":
                    aa_mass += deepnovo_config.mass_AA["C(Carbamidomethylation)"]
                else:
                    aa_mass += deepnovo_config.mass_AA[tag[ti+s]]
                new_socre.append(tag_score[ti+s])
                score += tag_score[ti+s]

            score = float(score) / Wsize

            if direction == 0 or direction == "0":
                left_mass = left - aa_mass
                right_mass = right
            else:
                right_mass = right - aa_mass
                left_mass = left

            if new_tag not in tag_list:
                if score < -1.0:
                    continue
                new_score_ls = [str(x) for x in new_socre]

                tag_score_list.append(score)
                tag_list.append(new_tag)
                tagResulet_ls.append(TagResualt(str(direction),
                                                new_tag,
                                                str(score),
                                                new_score_ls,
                                                left_mass,
                                                right_mass
                                                ))
        except(IndexError):
            break
    return tagResulet_ls,tag_list,tag_score_list



def remove_bracket_content(s):
    return re.sub(r'\(.*?\)', '', s)

def Test_GCNTag(args):
    moz_dict, instensity_dict = read_mgf(args.denovo_input_spectrum_file,args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_path = 'model3.pth'
    model = torch.load(model_save_path)
    model.eval()
    tag_list = []
    tag_score_list = []

    tag_list2 = []
    tag_score_list2 = []
    with open("./PXD004565/PXD004565_seek(1)random10000_IF_PreProcess_0_select10000_peaknum_250_nomod_filter3_0.4_3.tagbeamsearch","w") as fw:
        with open("./PXD004565/PXD004565_seek(1)random10000_IF_PreProcess_0_select10000_peaknum_250_nomod.tagbeamsearch","r") as f:
            content = f.readlines()
            for idx, line in enumerate(content):
                if line.startswith("BEGIN"):
                    fw.write(line)
                    scan_no = content[idx + 1].split("\t")[0]
                    scan_no_idx = idx + 1
                    cont = 0
                    tag_list = []
                    tag_score_list = []
                    tagResualt_list = []
                    cout = 0


                elif scan_no_idx == idx:
                    fw.write(line)
                    line_ls = line.split("\t")
                    scan = line_ls[0]
                elif line[0].isdigit():
                    # print(scan_no)

                    line_ls = line.split("\t")
                    tag_sequece = line_ls[2]
                    # print("tagseunce:",tag_sequece)
                    tag_sequece = remove_bracket_content(tag_sequece)
                    # print("tagseunce:", tag_sequece)
                    if len(tag_sequece) < 3:
                        continue

                    tag_sequece = tag_sequece.replace(",", "")
                    ls1 = [0 for x in range(40)]
                    ls2 = [0 for x in range(40)]
                    for aidx, aa in enumerate(tag_sequece):
                        if aa == "C":
                            aa = "C(Carbamidomethylation)"
                        aa_id = deepnovo_config.vocab_reverse.index(aa)
                        # print(aidx,tag)
                        ls1[aidx] = aa_id

                    tag_aa_score_ls = [float(x) for x in line_ls[4].split(",")]
                    for s_idx, score in enumerate(tag_aa_score_ls):
                        ls2[s_idx] = score

                    direction = line_ls[1]
                    tagscore = line_ls[3]
                    left_mass = float(line_ls[5])
                    rigth_mass = float(line_ls[6][:-1])
                    moz = moz_dict[scan]
                    instensity =instensity_dict[scan]

                    tag = np.array(ls1)
                    score_n = np.array(ls2)
                    x = np.append(tag,score_n)
                    x = np.append(x,moz)
                    x = np.append(x,instensity)
                    x = torch.from_numpy(x)
                    x= x.to(device)
                    x = x.to(torch.float32)
                    # print(x.size)
                    # x = x.view(1,x.shape)

                    x = x.view(1,x.size(0))
                    # print(x.shape)
                    out = model(x)
                    # print(out)
                    pred = torch.tensor([[1] if num[0] >= 0.4 else [0] for num in out]).to(device)

                    if pred == 1:
                        # tag_position_ls = [str(x) for x in tag_aa_score_ls]
                        tag_list.append(tag_sequece)
                        # tag_score_list.append(tagscore)
                        #
                        # tagResualt_list.append(TagResualt(direction,
                        #                                   tag_sequece,
                        #                                   tagscore,
                        #                                   tag_position_ls,
                        #                                   str(left_mass),
                        #                                   str(rigth_mass)
                        #                                   ))
                        cout += 1
                        fw.write(line)
                    else:
                        if len(tag) < 4:
                            continue
                        else:
                            tagResualt_list, tag_list, tag_score_list = get_fixedTag(tag_sequece,tag_aa_score_ls,tag_list,tag_score_list,3, left_mass,rigth_mass,direction,tagResualt_list)
                        # print("len", len(index_list), len(tagResualt_list))

                elif line.startswith("END"):

                    tag_score_list_np = np.array(tag_score_list)
                    index_list = np.argsort(tag_score_list_np)[::-1]
                    # print("len",len(index_list),len(tagResualt_list),tag_list,tagResualt_list)
                    for id, listidx in enumerate(index_list):
                        if id >= 100-cout:
                            break
                        position_score =""


                        for ps in tagResualt_list[listidx].tag_position_score:
                            position_score += ps
                        # print("tagResualt_list[listidx].tag_seq", tagResualt_list[listidx].tag_seq)
                        tag_rline = str(id) + "\t"
                        tag_rline += str(tagResualt_list[listidx].direction)+"\t"
                        tag_rline += tagResualt_list[listidx].tag_seq +"\t"
                        tag_rline += str(tagResualt_list[listidx].tag_score) + "\t"
                        tag_rline += position_score + "\t"
                        tag_rline += str(tagResualt_list[listidx].left) +"\t"
                        tag_rline += str(tagResualt_list[listidx].rigth)+"\n"
                        fw.write(tag_rline)


                        #
                        # line_tag = "\t".join([str(id),
                        #                str(tagResualt_list[listidx].direction),
                        #                tagResualt_list[listidx].tag_seq,
                        #                str(tagResualt_list[listidx].tag_score),
                        #                position_score,
                        #                str(tagResualt_list[listidx].left),
                        #                str(tagResualt_list[listidx].rigth)+"/n"])

                else:
                    fw.write(line)






if __name__ == '__main__':
    param_path = "./param/cross.9high_80k.exclude_bacillus_4_PeakNetwork_NH3H2O_InternalIons_Edge.cfg"
    if os.path.isfile(param_path):
        dir, param_file = os.path.split(param_path)
        # log_file_name = "top5_" + param_file[-4] + ".log"
        now = datetime.datetime.now().strftime("%Y%m%d%H%M")
        args = init_args(param_path)

        if os.path.exists(args.train_dir):
            pass
        else:
            os.makedirs(args.train_dir)
        Test_GCNTag(args)

    elif os.path.isdir(param_path):
        list_dir = os.listdir(param_path)
        list_dir.sort(key=lambda x: int(x[33]))
        print(list_dir)
        for file in list_dir:
            one_param_path = os.path.join(param_path, file)
            if os.path.isfile(one_param_path):
                now = datetime.datetime.now().strftime("%Y%m%d%H%M")
                args = init_args(one_param_path)

                if os.path.exists(args.train_dir):
                    pass
                else:
                    os.makedirs(args.train_dir)
                Test_GCNTag(args)