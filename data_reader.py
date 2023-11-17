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

def to_tensor(data_dict: dict) -> dict:
    temp = [(k, torch.from_numpy(v)) for k, v in data_dict.items()]
    return dict(temp)


def pad_to_length(input_data: list, pad_token, max_length: int) -> list:
    assert len(input_data) <= max_length
    result = input_data[:]
    for i in range(max_length - len(result)):
        result.append(pad_token)
    return result


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
class DenovoData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    precursormass: float
    spectrum_representation: np.ndarray
    original_dda_feature: DDAFeature

@dataclass
class MGFfeature:
    PEPMASS: float
    CHARGE: int
    SCANS: str
    SEQ: str
    RTINSECONDS: float
    MOZ_LIST: list
    INTENSITY_LIST: list

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


class DeepNovoTrainDataset(Dataset):
    def __init__(self, feature_filename, spectrum_filename, args, transform=None):
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
        # read spectrum location file
        spectrum_location_file = spectrum_filename + '.location.pytorch.pkl'
        if os.path.exists(spectrum_location_file):
            logger.info(f"read cached spectrum locations")
            with open(spectrum_location_file, 'rb') as fr:
                self.spectrum_location_dict = pickle.load(fr)
            # print("124 spectrum_location_dict",self.spectrum_location_dict)
        else:
            logger.info("build spectrum location from scratch")
            spectrum_location_dict = {}
            line = True
            with open(spectrum_filename, 'r') as f:
                while line:
                    # print(line)
                    current_location = f.tell()
                    line = f.readline()
                    if "BEGIN IONS" in line:
                        spectrum_location = current_location
                    elif "SCANS=" in line:
                        scan = re.split('[=\r\n]', line)[1]
                        # print(scan)
                        # print(spectrum_location)
                        spectrum_location_dict[scan] = spectrum_location
            self.spectrum_location_dict = spectrum_location_dict
            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(self.spectrum_location_dict, fw)
            # print("141 spectrum_location_dict",self.spectrum_location_dict)
            # with open(spectrum_filename, "r")as f:
            #     data = f.read().split("\n")
            # for line in data:
            #     if line:
            #         if line.startswith("TITLE"):
            #             moz_list = []
            #             intensity_list = []
            #             pass
            #         elif line.startswith("PEPMASS"):
            #             pepmass = float(line.split("=")[1])
            #         elif line.startswith("CHARGE"):
            #             charge = int(line.split("=")[1][0])
            #         elif line.startswith("RTINSECONDS"):
            #             rt = float(line.split("=")[1])
            #         elif line.startswith("SCANS"):
            #             scans = line.split("=")[1]
            #         elif line.startswith("SEQ"):
            #             seq = line.split("=")[1]
            #         elif line.startswith("BEGIN IONS"):
            #             pass
            #         elif line.startswith("END IONS"):
            #             mgffeature = MGFfeature(PEPMASS=pepmass,
            #                                     CHARGE=charge,
            #                                     RTINSECONDS=rt,
            #                                     SCANS=scans,
            #                                     SEQ=seq,
            #                                     MOZ_LIST=moz_list,
            #                                     INTENSITY_LIST=intensity_list)
            #             spectrum_location_dict[scans] = mgffeature
            #         else:
            #             moz, intensity = line.split(" ")
            #             moz_list.append(float(moz))
            #             intensity_list.append(math.sqrt(float(intensity)))
            # self.spectrum_location_dict = spectrum_location_dict
            # with open(spectrum_location_file, 'wb') as fw:
            #     pickle.dump(self.spectrum_location_dict, fw)
        # read feature file
        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        with open(feature_filename, 'r') as fr:
            reader = csv.reader(fr, delimiter=',')
            header = next(reader)
            feature_id_index = header.index(deepnovo_config.col_feature_id)
            mz_index = header.index(deepnovo_config.col_precursor_mz)
            z_index = header.index(deepnovo_config.col_precursor_charge)
            rt_mean_index = header.index(deepnovo_config.col_rt_mean)
            seq_index = header.index(deepnovo_config.col_raw_sequence)
            scan_index = header.index(deepnovo_config.col_scan_list)
            feature_area_index = header.index(deepnovo_config.col_feature_area)
            for line in reader:
                # print(line)
                mass = (float(line[mz_index]) - deepnovo_config.mass_H) * float(line[z_index])
                ok, peptide = parse_raw_sequence(line[seq_index])
                if not ok:
                    skipped_by_ptm += 1
                    logger.debug(f"{line[seq_index]} skipped by ptm")
                    continue
                if mass > self.args.MZ_MAX:
                    skipped_by_mass += 1
                    logger.debug(f"{line[seq_index],mass} skipped by mass")
                    continue
                if len(peptide) >= self.args.MAX_LEN:
                    skipped_by_length += 1
                    logger.debug(f"{line[seq_index]} skipped by length")
                    continue
                new_feature = DDAFeature(feature_id=line[feature_id_index],
                                         mz=float(line[mz_index]),
                                         z=float(line[z_index]),
                                         rt_mean=float(line[rt_mean_index]),
                                         peptide=peptide,
                                         scan=line[scan_index],
                                         mass=mass,
                                         feature_area=line[feature_area_index])
                self.feature_list.append(new_feature)
        logger.info(f"read {len(self.feature_list)} features, {skipped_by_mass} skipped by mass, "
                    f"{skipped_by_ptm} skipped by unknown modification, {skipped_by_length} skipped by length")

    def __len__(self):
        return len(self.feature_list)

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


    def _get_feature(self, feature: DDAFeature) -> TrainData:
        spectrum_location = self.spectrum_location_dict[feature.scan]
        self.input_spectrum_handle.seek(spectrum_location)
        # parse header lines
        line = self.input_spectrum_handle.readline()
        assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
        line = self.input_spectrum_handle.readline()
        assert "TITLE=" in line, "Error: wrong input TITLE="
        line = self.input_spectrum_handle.readline()
        assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
        line = self.input_spectrum_handle.readline()
        assert "CHARGE=" in line, "Error: wrong input CHARGE="
        line = self.input_spectrum_handle.readline()
        assert "SCANS=" in line, "Error: wrong input SCANS="
        line = self.input_spectrum_handle.readline()
        assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
        line = self.input_spectrum_handle.readline()
        assert "SEQ=" in line, "Error: wrong input SEQ="

        mz_list, intensity_list = self._parse_spectrum_ion()

        # mz_list = mgffeature.MOZ_LIST
        # intensity_list = mgffeature.INTENSITY_LIST
        peak_location, peak_intensity, spectrum_representation = process_peaks(mz_list, intensity_list, feature.mass, self.args)

        assert np.max(peak_intensity) < 1.0 + 1e-5

        # print("feature.peptide",feature.peptide,feature.scan)

        peptide_id_list = [deepnovo_config.vocab[x] for x in feature.peptide]
        forward_id_input = [deepnovo_config.GO_ID] + peptide_id_list
        forward_id_target = peptide_id_list + [deepnovo_config.EOS_ID]
        forward_ion_location_index_list = []
        prefix_mass = 0.
        for i, id in enumerate(forward_id_input):
            prefix_mass += deepnovo_config.mass_ID[id]
            ion_location = get_ion_index(feature.mass, prefix_mass, forward_id_input[:i+1], 0, args=self.args)
            forward_ion_location_index_list.append(ion_location)

        backward_id_input = [deepnovo_config.EOS_ID] + peptide_id_list[::-1]
        backward_id_target = peptide_id_list[::-1] + [deepnovo_config.GO_ID]
        backward_ion_location_index_list = []
        suffix_mass = 0
        for i, id in enumerate(backward_id_input):
            suffix_mass += deepnovo_config.mass_ID[id]
            ion_location = get_ion_index(feature.mass, suffix_mass,backward_id_input[:i+1], 1, args=self.args)
            backward_ion_location_index_list.append(ion_location)

        return TrainData(peak_location=peak_location,
                         peak_intensity=peak_intensity,
                         precursormass=feature.mass,
                         spectrum_representation=spectrum_representation,
                         forward_id_target=forward_id_target,
                         backward_id_target=backward_id_target,
                         forward_ion_location_index_list=forward_ion_location_index_list,
                         backward_ion_location_index_list=backward_ion_location_index_list,
                         forward_id_input=forward_id_input,
                         backward_id_input=backward_id_input)

    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = open(self.spectrum_filename, 'r')
        feature = self.feature_list[idx]
        return self._get_feature(feature)


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
    train_data_list.sort(key=lambda x: len(x.forward_id_target), reverse=True)
    batch_max_seq_len = len(train_data_list[0].forward_id_target)
    ion_index_shape = train_data_list[0].forward_ion_location_index_list[0].shape
    # print(ion_index_shape, deepnovo_config.vocab_size, deepnovo_config.num_ion)
    # assert ion_index_shape == (deepnovo_config.vocab_size, deepnovo_config.num_ion)

    peak_location = [x.peak_location for x in train_data_list]
    peak_location = np.stack(peak_location) # [batch_size, N]
    peak_location = torch.from_numpy(peak_location)

    peak_intensity = [x.peak_intensity for x in train_data_list]
    peak_intensity = np.stack(peak_intensity) # [batch_size, N]
    peak_intensity = torch.from_numpy(peak_intensity)

    spectrum_representation = [x.spectrum_representation for x in train_data_list]
    spectrum_representation = np.stack(spectrum_representation)  # [batch_size, embed_size]
    spectrum_representation = torch.from_numpy(spectrum_representation)

    batch_forward_ion_index = []
    batch_forward_id_target = []
    batch_forward_id_input = []
    batch_precursormass = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                               np.float32)
        forward_ion_index = np.stack(data.forward_ion_location_index_list)
        ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
        batch_forward_ion_index.append(ion_index)

        f_target = np.zeros((batch_max_seq_len,), np.int64)
        forward_target = np.array(data.forward_id_target, np.int64)
        f_target[:forward_target.shape[0]] = forward_target
        batch_forward_id_target.append(f_target)

        f_input = np.zeros((batch_max_seq_len,), np.int64)
        forward_input = np.array(data.forward_id_input, np.int64)
        f_input[:forward_input.shape[0]] = forward_input
        batch_forward_id_input.append(f_input)

        batch_precursormass.append(data.precursormass)

    batch_forward_id_target = torch.from_numpy(np.stack(batch_forward_id_target))  # [batch_size, T]
    batch_forward_ion_index = torch.from_numpy(np.stack(batch_forward_ion_index))  # [batch, T, 26, 8]
    batch_forward_id_input = torch.from_numpy(np.stack(batch_forward_id_input))
    batch_precursormass = torch.from_numpy(np.array(batch_precursormass))

    batch_backward_ion_index = []
    batch_backward_id_target = []
    batch_backward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                             np.float32)
        backward_ion_index = np.stack(data.backward_ion_location_index_list)
        ion_index[:backward_ion_index.shape[0], :, :] = backward_ion_index
        batch_backward_ion_index.append(ion_index)

        b_target = np.zeros((batch_max_seq_len,), np.int64)
        backward_target = np.array(data.backward_id_target, np.int64)
        b_target[:backward_target.shape[0]] = backward_target
        batch_backward_id_target.append(b_target)

        b_input = np.zeros((batch_max_seq_len,), np.int64)
        backward_input = np.array(data.backward_id_input, np.int64)
        b_input[:backward_input.shape[0]] = backward_input
        batch_backward_id_input.append(b_input)

    batch_backward_id_target = torch.from_numpy(np.stack(batch_backward_id_target))  # [batch_size, T]
    batch_backward_ion_index = torch.from_numpy(np.stack(batch_backward_ion_index))  # [batch, T, 26, 8]
    batch_backward_id_input = torch.from_numpy(np.stack(batch_backward_id_input))

    return (peak_location,
            peak_intensity,
            batch_precursormass,
            spectrum_representation,
            batch_forward_id_target,
            batch_backward_id_target,
            batch_forward_ion_index,
            batch_backward_ion_index,
            batch_forward_id_input,
            batch_backward_id_input
            )


# helper functions
def chunks(l, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]


@dataclass
class CMS2Spectrum:
    LIST_PRECURSOR_CHARGE:list # 母离子电荷信息
    LIST_PRECURSOR_MOZ: list
    LIST_PEAK_MOZ:list
    LIST_PEAK_INT:list


class DeepNovoDenovoDataset(DeepNovoTrainDataset):
    # override _get_feature method
    def _get_feature(self, feature: DDAFeature) -> DenovoData:
        # print("spectrum_location_dict", self.spectrum_location_dict)
        spectrum_location = self.spectrum_location_dict[feature.scan]
        self.input_spectrum_handle.seek(spectrum_location)
        # parse header lines
        line = self.input_spectrum_handle.readline()
        assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
        line = self.input_spectrum_handle.readline()
        assert "TITLE=" in line, "Error: wrong input TITLE="
        line = self.input_spectrum_handle.readline()
        # print("line:", line,line[-1],line[-2])
        pepmass = line[:-1].split('=')[1]
        # print("pepmass",pepmass)
        assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
        line = self.input_spectrum_handle.readline()
        charge = line[:-1].split('=')[1]
        # print("chage:",charge)
        assert "CHARGE=" in line, "Error: wrong input CHARGE="
        line = self.input_spectrum_handle.readline()
        assert "SCANS=" in line, "Error: wrong input SCANS="
        line = self.input_spectrum_handle.readline()
        assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
        line = self.input_spectrum_handle.readline()
        assert "SEQ=" in line, "Error: wrong input SEQ="

        # mz_list, intensity_list = self._parse_spectrum_ion()
        # print("charge:",charge,type(charge),"2" in charge)
        if self.args.IF_PreProcess == 0:
            mz_list, intensity_list = self._parse_spectrum_ion()
        else:
            if "2" in charge:
                mz_list, intensity_list = self._parse_spectrum_ion()
            else:
                mz_list, intensity_list = self._parse_spectrum_ion_tag()

                dataMS2Spectrum = CMS2Spectrum(LIST_PRECURSOR_CHARGE = [int(charge)],
                                               LIST_PRECURSOR_MOZ = [float(pepmass)],
                                               LIST_PEAK_MOZ = mz_list,
                                               LIST_PEAK_INT = intensity_list)

                # print("moz:",type(mz_list),len(mz_list), mz_list)
                # print("int:", type(intensity_list),len(intensity_list),intensity_list)
                # print("feature.mass:", feature.mass)

                # 数据预处理进行同位素峰簇的转换以及母离子高电荷向低电荷之间的转换
                self.__soldierGetSingleChargePeaksMS2TESTING(dataMS2Spectrum)
                # mz_list = mgffeature.MOZ_LIST
                # intensity_list = mgffeature.INTENSITY_LIST
                mz_list = dataMS2Spectrum.LIST_PEAK_MOZ
                intensity_list = [math.sqrt(intensity) for intensity in dataMS2Spectrum.LIST_PEAK_INT]
        # print("moz:", type(mz_list), len(mz_list), mz_list)
        # print("int:", type(intensity_list), len(intensity_list), intensity_list)

        peak_location, peak_intensity, spectrum_representation = process_peaks(mz_list, intensity_list, feature.mass, self.args)

        return DenovoData(peak_location=peak_location,
                          peak_intensity=peak_intensity,
                          precursormass=feature.mass,
                          spectrum_representation=spectrum_representation,
                          original_dda_feature=feature)


    def __soldierGetSingleChargePeaksMS2TESTING(self, dataMS2Spectrum):

        # print("before line 508: moz intensiry ", dataMS2Spectrum.LIST_PRECURSOR_MOZ)
        # print("mzo:",dataMS2Spectrum.LIST_PEAK_MOZ)
        # print("int:",dataMS2Spectrum.LIST_PEAK_INT)

        # check
        delete_index_list = []
        for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
            if dataMS2Spectrum.LIST_PEAK_MOZ[i] < 1.00727645224:  # MASS_PROTON_MONO
                delete_index_list.append(i)
            else:
                break
        if delete_index_list:
            for index in delete_index_list[::-1]:
                dataMS2Spectrum.LIST_PEAK_MOZ.pop(index)
                dataMS2Spectrum.LIST_PEAK_INT.pop(index)
            delete_index_list = []

        max_charge = max(dataMS2Spectrum.LIST_PRECURSOR_CHARGE)

        new_peak_int = []
        new_peak_moz = []
        isotopic_tol = [1.0025 / c for c in range(1, max_charge + 1)]  # MASS_NEUTRON_AVRG = 1.0025
        # 1.0030, 0.5015, 0.3334, 0.25075, 0.2016...

        # cluster_start_counter = 0
        charge_state = -1  # 独峰

        while 0 < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            cluster_index_list = []
            # 小于isotopic_tol[-1] - tol 就check下一谱峰
            # 检查开始列表中的任意电荷状态的距离，由index可得电荷状态index+1
            # 超过isotopic_tol[0]就进行归并，并跳转下一峰
            # index = cluster_start_counter
            max_int = -1
            max_int_peak_index = -1
            for i in range(len(dataMS2Spectrum.LIST_PEAK_MOZ)):
                if dataMS2Spectrum.LIST_PEAK_INT[i] > max_int:
                    max_int = dataMS2Spectrum.LIST_PEAK_INT[i]
                    max_int_peak_index = i
            # 得到最高峰信号的地址了

            # 而后左右检测是否可有峰簇

            # while tmp_check_index < len(dataMS2Spectrum.LIST_PEAK_MOZ):
            #     if -1 == charge_state:
            #         pass
            # ########################################################
            # not complete

            # -------------------------- right -----------------------------
            # ##############################################################
            tmp_check_index = max_int_peak_index
            # index = 0
            cluster_index_list.append(tmp_check_index)
            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
            tmp_check_index += 1
            # charge == -1: cluster is not complete

            while tmp_check_index < len(dataMS2Spectrum.LIST_PEAK_MOZ):

                peak_tolerance = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index] - cluster_tail_moz
                """
                
                
                if self.msms_ppm == 1:
                    ppm_tol_da = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index] * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol
                """
                ppm_tol_da = self.args.TOL
                prep_tolerance = ppm_tol_da
                if peak_tolerance < isotopic_tol[-1] - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index += 1

                elif peak_tolerance > isotopic_tol[0] + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:

                    # 先确定电荷状态，再继续cluster构建
                    if charge_state == -1:
                        for tol_index in range(len(isotopic_tol)):
                            if math.isclose(peak_tolerance, isotopic_tol[tol_index], abs_tol=prep_tolerance):
                                charge_state = tol_index + 1
                                # 确定质荷比信息
                                cluster_index_list.append(tmp_check_index)
                                cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                                break
                        if charge_state == -1:
                            # 继续向后寻找（MUST）
                            tmp_check_index += 1
                        else:

                            # 连续向后构造
                            tmp_check_index += 1


                    # 已经确定电荷状态
                    else:
                        # 仍然是我的同位素峰，进入峰簇下标列表
                        if math.isclose(peak_tolerance, isotopic_tol[charge_state - 1], abs_tol=prep_tolerance):

                            # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                            nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[-1]]
                            if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] > 1.2 * nearest_int:
                                tmp_check_index += 1
                                continue
                            cluster_index_list.append(tmp_check_index)
                            cluster_tail_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                            tmp_check_index += 1
                        # 超过了枚举电荷的范围，break
                        elif peak_tolerance - isotopic_tol[charge_state - 1] > prep_tolerance:
                            break
                        # 又不匹配，又不越界，啥也不是，继续走吧
                        else:
                            tmp_check_index += 1
            # while index < len(dataMS2Spectrum.LIST_PEAK_MOZ) over.

            # -------------------------- left ------------------------------
            # ##############################################################
            tmp_check_index = max_int_peak_index - 1
            cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[max_int_peak_index]
            while tmp_check_index >= 0 and dataMS2Spectrum.LIST_PEAK_MOZ:

                peak_tolerance = cluster_left_moz - dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                """
                                if self.msms_ppm == 1:
                    ppm_tol_da = cluster_left_moz * self.msms_fraction
                else:
                    ppm_tol_da = self.msms_tol
                """
                ppm_tol_da = self.args.TOL
                prep_tolerance = ppm_tol_da


                if peak_tolerance < isotopic_tol[-1] - prep_tolerance:
                    # 小于最小,跳过去
                    # 要继续找，不要break
                    tmp_check_index -= 1

                elif peak_tolerance > isotopic_tol[0] + prep_tolerance:
                    # 大于最大
                    # 不要继续找，要break
                    break

                else:

                    # 先确定电荷状态，再继续cluster构建
                    if charge_state == -1:
                        for tol_index in range(len(isotopic_tol)):
                            if math.isclose(peak_tolerance, isotopic_tol[tol_index], abs_tol=prep_tolerance):

                                # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                                # nearest_int: cluster list中最左端的谱峰信号的强度值
                                nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]
                                # if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.2 * nearest_int:
                                if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.3 * nearest_int:
                                    # break
                                    tmp_check_index -= 1
                                    continue

                                charge_state = tol_index + 1
                                # 确定质荷比信息
                                # cluster_index_list.append(tmp_check_index)
                                cluster_index_list = [tmp_check_index] + cluster_index_list
                                cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                                break
                        if charge_state == -1:
                            # 继续向前寻找（MUST）
                            tmp_check_index -= 1
                        else:

                            # 连续向后构造
                            tmp_check_index -= 1


                    # 已经确定电荷状态
                    else:
                        # 仍然是我的同位素峰，进入峰簇下标列表
                        if math.isclose(peak_tolerance, isotopic_tol[charge_state - 1], abs_tol=prep_tolerance):
                            # 尽可能避免误匹配的发生,混合谱峰时，拒绝构造高低错落类型谱峰
                            nearest_int = dataMS2Spectrum.LIST_PEAK_INT[cluster_index_list[0]]
                            # if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.2 * nearest_int:
                            if dataMS2Spectrum.LIST_PEAK_INT[tmp_check_index] < 0.4 * nearest_int:
                                # break
                                tmp_check_index -= 1
                                continue
                            cluster_index_list = [tmp_check_index] + cluster_index_list
                            cluster_left_moz = dataMS2Spectrum.LIST_PEAK_MOZ[tmp_check_index]
                            tmp_check_index -= 1
                        # 超过了枚举电荷的范围，break
                        elif peak_tolerance - isotopic_tol[charge_state - 1] > prep_tolerance:
                            break
                        # 又不匹配，又不越界，啥也不是，继续走吧
                        else:
                            tmp_check_index -= 1

            # --------------------------------------------------------------

            # cluster
            # cluster 构造结束，开始收工
            if charge_state == -1:
                # 删除，添加
                add_moz = dataMS2Spectrum.LIST_PEAK_MOZ.pop(cluster_index_list[0])
                add_int = dataMS2Spectrum.LIST_PEAK_INT.pop(cluster_index_list[0])
                new_peak_moz.append(add_moz)
                new_peak_int.append(add_int)
            else:

                # ##########################################
                # 20210809 ATTENTION ATTENTION ATTENTION ###
                # pop时，一定一定一定要注意地址的问题！ ####
                # ##########################################
                # --------------- QUESTION --------------- #
                # 这里要把mono峰pop出来，但是一定要放在最后#
                # 否则会改变其他谱峰的地址，导致pop峰错误  #
                # 再就是强度和质荷比错开了，导致tag提取错误#
                # ##########################################
                add_int = 0
                for buf_index in reversed(cluster_index_list[1:]):
                    try:
                        # dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                        # add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)

                        dataMS2Spectrum.LIST_PEAK_MOZ.pop(buf_index)
                        add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(buf_index)

                    except:
                        pass

                # 把mono峰的moz提出放在最后，包括mono对应的强度部分也是
                # 如果希望对add的强度做一些操作，可以在for loop try里头去整
                add_moz = dataMS2Spectrum.LIST_PEAK_MOZ.pop(cluster_index_list[0]) * charge_state
                add_moz -= (charge_state - 1) * 1.00727645224 # MASS_PROTON_MONO
                add_int += dataMS2Spectrum.LIST_PEAK_INT.pop(cluster_index_list[0])

                new_peak_moz.append(add_moz)
                new_peak_int.append(add_int)

            charge_state = -1
            # cluster 处理结束
        # while over

        # -------------------------------------------------------------------

        # 排序，检测合并
        index_order = np.argsort(new_peak_moz)
        new_peak_moz = [new_peak_moz[index] for index in index_order]
        new_peak_int = [new_peak_int[index] for index in index_order]

        # -------------------------------------------------------------------

        output_peak_moz = []
        output_peak_int = []

        # 下方：合并转换的谱峰，有可能会影响精度
        add_moz = 0
        add_int = 0
        i = 0
        jump_set = set()
        while i < len(new_peak_moz) - 1:
            add_moz = new_peak_moz[i]
            add_int = new_peak_int[i]

            if i in jump_set:
                i += 1
                continue

            for j in range(i + 1, len(new_peak_moz)):
                """
                if self.msms_ppm == 1:

                    prep_tolerance = new_peak_moz[j] * self.msms_fraction
                else:
                    prep_tolerance = self.msms_tol
                """
                prep_tolerance = self.args.TOL
                # prep_tolerance = new_peak_moz[j] * 2e-5

                if abs(new_peak_moz[i] - new_peak_moz[j]) < prep_tolerance:
                    add_moz = add_moz * add_int + new_peak_moz[j] * new_peak_int[j]
                    add_int += new_peak_int[j]
                    add_moz /= add_int
                    i = j
                    jump_set.add(j)
                # 仅看左右最远的0.02，两两之间就不看了
                elif abs(new_peak_moz[i] - new_peak_moz[j]) >= prep_tolerance:
                    output_peak_moz.append(add_moz)
                    output_peak_int.append(add_int)
                    i = j
                    break

            # if abs(new_peak_moz[i] - new_peak_moz[-1]) < prep_tolerance:
            #     output_peak_moz.append(add_moz)
            #     output_peak_int.append(add_int)
            #     i = j
            #     break

        if add_moz in output_peak_moz:
            pass
        else:
            output_peak_moz.append(add_moz)
            output_peak_int.append(add_int)

        # 活久见，图里没谱峰哒，就算辽叭
        if len(new_peak_moz) == 0:
            pass
        # 检测一下最后一根谱峰是否是可以添加的信息
        elif jump_set:
            if max(jump_set) == len(new_peak_moz):
                pass
            else:
                output_peak_moz.append(new_peak_moz[-1])
                output_peak_int.append(new_peak_int[-1])
        else:
            output_peak_moz.append(new_peak_moz[-1])
            output_peak_int.append(new_peak_int[-1])

        # 上方：合并转换的谱峰，有可能会影响精度
        # output_peak_moz = new_peak_moz
        # output_peak_int = new_peak_int

        dataMS2Spectrum.LIST_PEAK_MOZ = output_peak_moz
        dataMS2Spectrum.LIST_PEAK_INT = output_peak_int
        # print("after line821: moz intensiry", dataMS2Spectrum.LIST_PRECURSOR_MOZ)
        # print("moz:",dataMS2Spectrum.LIST_PEAK_MOZ)
        # print("int:",dataMS2Spectrum.LIST_PEAK_INT)

if __name__=='__main__':
    train_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_train,
                                     deepnovo_config.input_spectrum_file_train)
    print('finished')