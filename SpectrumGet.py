import torch
import numpy as np
mass_AA = {
           'A': 71.03711, # 0  3
           'R': 156.10111, # 1   4
           'N': 114.04293, # 2   5
           'D': 115.02694, # 3    7
           #~ 'C(Carbamidomethylation)': 103.00919, # 4
           'C(Carbamidomethylation)': 160.03065, # C(+57.02)  8
           # 'C(Carbamidomethylation)': 103.00919, # C(+57.02)  8
           #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           # 'C': 103.00919,
           'E': 129.04259, # 5   9
           'Q': 128.05858, # 6    10
           'G': 57.02146, # 7   12
           'H': 137.05891, # 8   13
           'I': 113.08406, # 9   14
           'L': 113.08406, # 10   15
           'K': 128.09496, # 11   16
           'M': 131.04049, # 12   17
           'F': 147.06841, # 13  19
           'P': 97.05276, # 14    20
           'S': 87.03203,  # 15   21
           'T': 101.04768,  # 16   22
           'W': 186.07931,  # 17   23
           'Y': 163.06333,  # 18  24
           'V': 99.06841,  # 19   25
           # 'M(Oxidation)': 147.0354,
           # 'Q(Deamidation)': 129.0426,
           # 'N(Deamidation)': 115.02695,
          }


# Da = 0.05


# 获得氨基酸质量映射 {710017:['A'],...}
def GetAAMap(Da):
    map = {}
    for aa in mass_AA:
        # if aa == "I":
        #     continue
        mass = mass_AA[aa]
        mass_min = mass - Da
        mass_max = mass + Da
        int_mass_min = int(mass_min * 1000)
        int_mass_max = int(mass_max * 1000)

        for i in range(int_mass_min,int_mass_max+1):
            if i in map:
                map[i].append(aa)
            else:
                map[i] = []
                map[i].append(aa)
    # print("map",map)
    return map


def GetDiffrenceMatrix(original_spectrum_tuple):
    # print(original_spectrum_tuple)
    peak_moz = original_spectrum_tuple[0].to("cpu")


    peak_moz_uns = peak_moz.unsqueeze(0)
    # peak_intensity = original_spectrum_tuple[1]
    # print("type(peak_moz)",type(peak_moz))
    # print("shape1",peak_moz_uns.shape)
    peak_moz_T = peak_moz_uns.T
    # print("shape2",peak_moz_T.shape)
    # print(peak_moz_uns.shape, peak_moz_T.shape)
    diffrence_Matrix = peak_moz_uns - peak_moz_T
    # print("diffrence_Matrix.shape",diffrence_Matrix.shape)
    diffrence_Matrix = torch.abs(diffrence_Matrix)
    # print(peak_moz)
    diffrence_Matrix_list = diffrence_Matrix.tolist()
    diffrence_Matrix_np = diffrence_Matrix.numpy()
    diffrence_Matrix_triu = np.triu(diffrence_Matrix_np,1)
    diffrence_Matrix_triu_list = diffrence_Matrix_triu.tolist()  # 返回的上三角矩阵
    # print("diffrence_Matrix", diffrence_Matrix_triu_list)
    # print(diffrence_Matrix)
    return diffrence_Matrix_triu_list

