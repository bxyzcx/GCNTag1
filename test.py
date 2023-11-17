# import numpy as np
# import csv
#
#
# pFind_res_dict = {}
# with open("/home/sdut-idea/Wuruitao/data/single/Ecoli_lys_5specise_valid.csv", "r")as f:
#     csv_reader = csv.reader(f)
#     next(csv_reader)
#     for line in csv_reader:
#         infolist = list(line)
#         pFind_res_dict[infolist[0]] = infolist
#
# print("all psm: ", len(pFind_res_dict))
#
# reverse_pFind_dict = {}
#
# for key, infolist in pFind_res_dict.items():
#     # print(infolist)
#     reverse_pFind_dict.setdefault(infolist[5], []).append([key, infolist[9]])
#
# print("all sequence: ", len(reverse_pFind_dict))
#
# sorted_reverse_pFind_dict = {}
#
# for sequence, filenamelist in reverse_pFind_dict.items():
#     score_list = [float(item[-1]) for item in filenamelist]
#     index_score_list_sort = np.argsort(np.array(score_list))
#     if len(filenamelist) > 3:
#         sorted_mirror_pairs_list = [filenamelist[i] for i in index_score_list_sort[:3]]
#     else:
#         sorted_mirror_pairs_list = [filenamelist[i] for i in index_score_list_sort]
#     sorted_reverse_pFind_dict[sequence] = sorted_mirror_pairs_list
#
# modification_dict = {"Carbamidomethyl[C]": "(+57.02)", "Oxidation[M]": "(+15.99)"}
# all_feature = []
# for peptide, filenamelist in sorted_reverse_pFind_dict.items():
#     for filename in filenamelist:
#         title, score = filename
#         # if try_title in try_title_dict.keys() and lys_title in lys_title_dict.keys():
#         try_sequence = pFind_res_dict[title][5]
#         modification = pFind_res_dict[title][10]
#         if len(try_sequence) <= 30:
#             if modification:
#                 sequence = ""
#                 moddict = {}
#                 mod_list = modification.split(";")[:-1]
#                 for mod in mod_list:
#                     index, name = mod.split(",")
#                     moddict[int(index)] = name
#                 for index, aa in enumerate(try_sequence):
#                     if index in moddict.keys():
#                         sequence += moddict[index]
#                     else:
#                         sequence += aa
#             else: sequence = try_sequence
#
#             try_EM = pFind_res_dict[title][2]
#             try_z = pFind_res_dict[title][3]
#             try_mz = str((float(try_EM) + (int(try_z) - 1) * 1.00727645224) / int(try_z))
#
#             feature = [title, try_mz, try_z, "0.0", sequence, title, "0.0:1.0", "1.0"]
#             all_feature.append(feature)
# print("all datasets: ", len(all_feature))
# with open("/home/sdut-idea/Wuruitao/data/single/new_Ecoli_lys_valid.csv", "w")as f:
#     f.write("spec_group_id,m/z,z,rt_mean,seq,scans,profile,feature area\n")
#     for feature in all_feature:
#         f.write(",".join(feature) + "\n")
import torch
import numpy as np

a = [1, 2, 3]
b = torch.from_numpy(np.array(a))
print(b.shape, b)
