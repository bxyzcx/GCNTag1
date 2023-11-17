import time
from dataclasses import dataclass

import SpectrumGet
from SpectrumGet import GetAAMap,GetDiffrenceMatrix
import os
import torch
import time
import logging
import deepnovo_config
import numpy as np
from dataclasses import dataclass
# from model import Direction, InferenceModelWrapper, device
from model_gcn import Direction, InferenceModelWrapper, device
from deepnovo_cython_modules import get_ion_index
from data_reader import DeepNovoDenovoDataset, chunks
from writer import BeamSearchedSequence, DenovoWriter, DDAFeature
import deepnovo_config
from enum import Enum
from TagWriter import TagWrite
logger = logging.getLogger(__name__)

class Direction(Enum):
    forward = 1
    backward = 2

@dataclass
class TagDAGNode:
    MASS: float  # 表示对应谱峰的质量
    INTENSITY: float  # 表示对应的谱峰强度
    IN: int  # 节点的入度信息
    OUT: int  # 节点的出度信息
    NUM_EDGE: int  # 边的个数
    LIST_EDGE: list  # 边的列表
    MOZ_INDEX: int  # 当前谱峰在谱图的位置
    FORHAEAD_MASS: float  # 前一个节点的质量，用于后续的
    NODE_POSITION:int   # 在所有的节点列表中的索引
    NOW_AA_LIST:list  # 到目前为止氨基酸的列表




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class TagDAGEdge:
    STAR_NODE_INDEX: int  # 此时边对应的起始节点在TagDAG中LIST_NODE中的索引是多少
    AA_LIST: list  # 此时这个边，也就是这个质量差对应的氨基酸的列表
    AA_LIST_SCORE: list  # 边对应氨基酸的打分
    END_NODE_INDEX: int  # 此时边对应的结束节点

    # WEIGHT_INT_AND_DEGREE = -1  # float  # 这里是半截儿打分，因为tolerance的打分需要实时算，所以就只有半截儿
    # TOL = -1                    # float  # 两峰间距，用于后续生成tag时MASS_TOLERANCE的计算
    # MASS_TOLERANCE = 100.0      # float  # 对应氨基酸的质量偏差，用于打分，目前没啥用。。但是留着吧
    # AA_MARK = ""                # int    # 这里是一个标记，可以对应氨基酸解释的集合，即该边对应的氨基酸、名称与质量信息的标记
    # LINK_NODE_INDEX = 1000      # int    # 有向边的边头，谱峰节点的下标，是个偏移地址


@dataclass
class TagDAG:
    LIST_NODE: list  # 所有Node的列表
    LIST_NODE_PEAK_INDEX: list  # 所有生成的NODE对应谱峰中的哪根峰
    NUM_NODE: int   # 节点个数
    IN_ZERO_NODE:list   # 入度为0的节点列表
    BACKORFORWORD :int # 0表示forward 1便是back
    #
    # LIST_NODE = []  # vector<CTagDAGNode>  # 存储CTagDAGNode，一个元素是一个节点
    # NUM_NODE = 0  # int  # 一共有多少个结点信息
    # IN_ZERO_NODE = []  # vector<int>  # 存储氨基酸路径的起点


class Graph(object):
    def __init__(self,args,original_spectrum_tuple,feature_dp):
        self.args = args
        self.aa_map = GetAAMap(self.args.TOL)
        self.diffrence_Matrix_triu_list = GetDiffrenceMatrix(original_spectrum_tuple)  # 上三角矩阵
        self.original_spectrum_tuple = original_spectrum_tuple
        # self.feature = feature_dp
        self.tagdag = []   # 有两个一个由forward构建的TagDAG，一个由backward构建的
        self.direction_cint_map = {Direction.forward: 0, Direction.backward: 1}
        self.peak_index_list = []   # 判断还有哪些谱峰没有被遍历到过,列表为0-499数字，刚好对应500峰的下标
        # self.peak_by_idx_list = [x for x in range(2,500)]
        self.peak_by_idx_list = [x for x in range(2, self.args.MAX_NUM_PEAK)]

        self.MIN_diff = 57.02694 - self.args.TOL - 1
        self.MAX_diff = 186.07931 + self.args.TOL + 1
        self.peak_moz_list = original_spectrum_tuple[0].tolist()
        # print("line82:",len(self.peak_moz_list),self.peak_moz_list)
        self.peak_intensity_list = original_spectrum_tuple[1].tolist()
        self.original_dda_feature = feature_dp.original_dda_feature

        self.precursor_mass = float(original_spectrum_tuple[2])
        # print("percursor_mass........:", self.precursor_mass)
        self.linshi_scorelist = []
        # print(self.precoresermass)
        # print(self.peak_moz_list)
        # print("self.aa_map",self.aa_map)


    def Create_Graph(self, model_wrapper,writer:TagWrite):
        start_time = time.time()
        # print("SpectrumGraph 100 Start Create_Graph ")
        start_node_tuple = self.__get_start_node__()
        # print(star_node_tuple)
        self.__get_graph__(start_node_tuple, model_wrapper)
        # print(" end get Graph start Writer_tag")
        # print("length of tagdag",len(self.tagdag))
        writer.__write_Tag__(self.original_dda_feature, self.tagdag, self.precursor_mass)


    @staticmethod
    def __get_start_node__() -> tuple:
        forward_start_node = TagDAGNode(MASS=0.0,
                                        INTENSITY=1.0,
                                        IN=0,
                                        OUT=0,
                                        NUM_EDGE=0,
                                        LIST_EDGE=[],
                                        MOZ_INDEX=0,
                                        FORHAEAD_MASS=0,
                                        NODE_POSITION=0,
                                        NOW_AA_LIST=[1]

        )
        backword_start_node = TagDAGNode(MASS=18.010563,
                                         INTENSITY=1.0,
                                         IN=0,
                                         OUT=0,
                                         NUM_EDGE=0,
                                         LIST_EDGE=[],
                                         MOZ_INDEX=1,
                                         FORHAEAD_MASS=0,
                                         NODE_POSITION=0,
                                         NOW_AA_LIST=[2]

        )

        return  forward_start_node,backword_start_node


    def __get_graph__(self, start_node_tuple,model_wrapper):

        for sp_idx, start_point in enumerate(start_node_tuple):
            if sp_idx == 0:
                direction = Direction.forward
                peak_i = 0  # 谱峰下标从0开始
                # print("forward blizi")
                # continue

            else:
                direction = Direction.backward
                peak_i = 1
                # print("backward ylizi")
                # continue


            tagdag = TagDAG(LIST_NODE=[start_point],
                            LIST_NODE_PEAK_INDEX=[peak_i],
                            NUM_NODE=1,
                            IN_ZERO_NODE=[start_point],
                            BACKORFORWORD = sp_idx)

            self.tagdag.append(tagdag)

            # peak_index_list = [x for x in range(2,500)]
            peak_index_list = [x for x in range(2,self.args.MAX_NUM_PEAK)]

            self.peak_index_list = peak_index_list

            self.__get_TagDAG__(direction, start_point, self.args, model_wrapper, 0)
            # print("end forward or backward next residual")
            # if peak_i == 0:
                # self.__get_TagDAG_Residual(direction, args, model_wrapper)
            # print("line157len(self.peak_index_list)", len(self.peak_by_idx_list))
            # print(self.peak_index_list)  # 还有剩余的多少根峰没有遍历到

        # print(len(self.tagdag),self.tagdag)
        # print(self.tagdag[0])
        # print(len(self.tagdag[0].LIST_NODE))
        # print("---------------------------------")
        # print(self.tagdag[1])
        # print(len(self.tagdag[1].LIST_NODE))
        Res_flag = False
        """
        for dag in self.tagdag:
            # print("run Res",len(dag.LIST_NODE))
            if len(dag.LIST_NODE)<5:
                Res_flag = True
        # print("run_flag",Res_flag)
        # if Res_flag:
            # print("Residual")
            # self.__get_TagDAG_Residual(direction, args, model_wrapper)
        """
        self.__get_TagDAG_Residual(direction, self.args, model_wrapper)
        # print("SpectraslGrahph 195 len tagdag",len(self.tagdag))
        # self.__get_TagDAG_Residual(direction, args, model_wrapper)
        # print("line157len(self.peak_index_list)", len(self.peak_by_idx_list))  # 还有剩余的多少根峰没有遍历到


    # 已知当前两根谱峰的差值
    def __getAALSIT__(self, diff):
        diffrence_int = int(diff * 1000)
        candidates_list = []

        if diffrence_int in self.aa_map:
            for aa in self.aa_map[diffrence_int]:
                candidates_list.append(aa)
        candidates_id_list = [deepnovo_config.vocab_reverse.index(x) for x in candidates_list]
        # if len(candidates_id_list) >=1:
        #     print("line 143,candidates_list,candidates_id_list",diff,diffrence_int,candidates_list,candidates_id_list)
        return candidates_id_list


    def __OP_model_input(self,block_ion_location) ->tuple:
        block_ion_location = torch.from_numpy(np.array(block_ion_location)).to(
            device)  # [batch, 26, 12] 后面开始会有 [5(新加入current_path_list为5)*4,26,12
        # print("block_ion_location.shape",block_ion_location.shape)
        block_ion_location = torch.unsqueeze(block_ion_location, dim=0)  # [batch, 1, 26, 12]  # 因为此时代码没有设batchsize因此为了可以正常输入进行两次Unsqueeze
        block_ion_location = torch.unsqueeze(block_ion_location, dim=0)  # [batch, 1, 26, 12]
        # print(ion_location)

        block_peak_location = self.original_spectrum_tuple[0]  # 当前batch的所有谱图
        block_peak_intensity = self.original_spectrum_tuple[1]
        block_precursormass = self.original_spectrum_tuple[2]

        block_peak_location = [block_peak_location]
        block_peak_intensity = [block_peak_intensity]
        block_precursormass = [block_precursormass]

        # print("block_peak_location",block_peak_location)
        block_ion_location = block_ion_location.contiguous()
        block_peak_location = torch.stack(block_peak_location, dim=0).contiguous()  # 真实谱图数据质荷比强度信息转为tensor
        block_peak_intensity = torch.stack(block_peak_intensity, dim=0).contiguous()
        block_precursormass = torch.stack(block_precursormass, dim=0).contiguous()


        block_state_tuple = None
        block_aa_id_input = None

        # print("block_ion_location", block_ion_location.shape)
        # print("block_ion_location", block_peak_location.shape)
        # print("block_ion_location", block_peak_intensity.shape)
        # print("block_ion_location", block_precursormass.shape)

        return block_ion_location,block_peak_location,block_peak_intensity,block_precursormass,block_aa_id_input,block_state_tuple


    def __get_TagDAG__(self, direction:Direction, now_point:TagDAGNode,args,model_wrapper,iiiii):

        ion_location = get_ion_index(self.precursor_mass, now_point.MASS, now_point.NOW_AA_LIST,
                                     self.direction_cint_map[direction], args=args)

        model_input = self.__OP_model_input(ion_location)
        current_log_prob, new_state_tuple = model_wrapper.step(model_input[0],
                                                               model_input[1],
                                                               model_input[2],
                                                               model_input[3],
                                                               model_input[4],
                                                               model_input[5],
                                                               direction)

        current_log_prob = current_log_prob.cpu().detach().numpy()
        # print("current_log_prob",current_log_prob)

        for idx_d,diffrence in enumerate(self.diffrence_Matrix_triu_list[now_point.MOZ_INDEX]):
            if diffrence == 0:
                continue
            if diffrence < self.MIN_diff:
                continue
            if diffrence > self.MAX_diff:
                break

            aa_list = self.__getAALSIT__(diffrence)

            if len(aa_list) >= 1:

                score_list = [float(current_log_prob[0][x]) for x in aa_list]


                if len(aa_list) > 1:
                    max_score = max(score_list)
                    score_ls = []
                    aa_ls = []
                    for idxx,x in enumerate(aa_list):
                        now_aa_score = float(current_log_prob[0][x])
                        if abs(now_aa_score-max_score)<= 3.5: # 连接某根谱峰时有多个氨基酸在列表里并且氨基酸数值相差较大的情况
                            score_ls.append(now_aa_score)
                            aa_ls.append(x)

                    score_list = score_ls
                    aa_list = aa_ls


                # print("scorelist",score_list)
                max_score_inlist = max(score_list)
                max_score_inlist_index = score_list.index(max_score_inlist)

                now_aa_list = now_point.NOW_AA_LIST
                if len(now_aa_list) < 7:
                    now_aa_list.append(aa_list[max_score_inlist_index])
                # print("now_aa_list",aa_listnow)

                # print(score_list)
                # 如果当前的谱峰已经生成过Node类型则不需要重新生成，只需要进行后续饿的信息更新即可
                # print("self.peak_intensity_list",type(self.peak_intensity_list),self.peak_intensity_list)
                if idx_d in self.peak_index_list:  # in self.peak_index_list 如果在这个列表里面则表示谱峰没有被遍历到过
                    if idx_d in self.peak_by_idx_list:
                        self.peak_by_idx_list.remove(idx_d)
                    node = TagDAGNode(MASS=self.peak_moz_list[idx_d],
                                      INTENSITY=self.peak_intensity_list[idx_d],
                                      IN=1,
                                      OUT=0,
                                      NUM_EDGE=0,
                                      LIST_EDGE=[],
                                      MOZ_INDEX = idx_d,
                                      FORHAEAD_MASS=now_point.MASS,
                                      NODE_POSITION=len(self.tagdag[-1].LIST_NODE),
                                      NOW_AA_LIST=now_aa_list

                    )
                    self.peak_index_list.remove(idx_d)
                    self.tagdag[-1].LIST_NODE.append(node)
                    self.tagdag[-1].LIST_NODE_PEAK_INDEX.append(idx_d)
                    self.tagdag[-1].NUM_NODE += 1

                    for score in score_list:
                        self.linshi_scorelist.append(float(score))
                    edge = TagDAGEdge(STAR_NODE_INDEX=now_point.NODE_POSITION,
                                      AA_LIST=aa_list,
                                      AA_LIST_SCORE=score_list,
                                      END_NODE_INDEX=node.NODE_POSITION
                                      )
                    now_point.OUT += 1
                    now_point.NUM_EDGE += 1
                    now_point.LIST_EDGE.append(edge)
                    self.__get_TagDAG__(direction, node, args, model_wrapper,iiiii)

                else:
                    # print("idx_d",idx_d,tagdag.LIST_NODE_PEAK_INDEX)
                    node_index = self.tagdag[-1].LIST_NODE_PEAK_INDEX.index(idx_d)
                    node = self.tagdag[-1].LIST_NODE[node_index]
                    node.IN += 1
                    edge = TagDAGEdge(STAR_NODE_INDEX=now_point.NODE_POSITION,
                                      AA_LIST=aa_list,
                                      AA_LIST_SCORE=score_list,
                                      END_NODE_INDEX=node.NODE_POSITION
                                      )
                    now_point.OUT += 1
                    now_point.NUM_EDGE += 1
                    now_point.LIST_EDGE.append(edge)

        # print("back")


    def __get_TagDAG_Residual(self, direction:Direction,args,model_wrapper):
        # peak_index_list = self.peak_by_idx_list
        # for i in self.peak_by_idx_list:
        #     print("without search moz:",self.original_spectrum_tuple[0][i])

        while True:
            # print("now length:",len(self.peak_by_idx_list))
            # self.peak_index_list = [x for x in range(2, 500)]
            self.peak_index_list = [x for x in range(2, self.args.MAX_NUM_PEAK)]
            if len(self.peak_by_idx_list) == 0:
                break
            # print(first_num)
            first_num = self.peak_by_idx_list[0]

            # print("line 317 ",first_num,self.peak_moz_list[first_num],self.peak_intensity_list[first_num],)
            # if  direction == Direction.forward:
            res_start_point = TagDAGNode(MASS=self.peak_moz_list[first_num],
                                        INTENSITY=self.peak_intensity_list[first_num],
                                        IN=0,
                                        OUT=0,
                                        NUM_EDGE=0,
                                        LIST_EDGE=[],
                                        MOZ_INDEX=first_num,
                                        FORHAEAD_MASS=0,
                                        NODE_POSITION=len(self.tagdag[-1].LIST_NODE),
                                        NOW_AA_LIST=[1])
            tagdag = TagDAG(LIST_NODE=[res_start_point],LIST_NODE_PEAK_INDEX=[first_num],NUM_NODE=1,IN_ZERO_NODE=[res_start_point],BACKORFORWORD=0)
            self.tagdag.append(tagdag)
            # else:
            #     res_start_point = TagDAGNode(MASS=self.peak_moz_list[first_num],
            #                                  INTENSITY=self.peak_intensity_list[first_num],
            #                                  IN=0,
            #                                  OUT=0,
            #                                  NUM_EDGE=0,
            #                                  LIST_EDGE=[],
            #                                  MOZ_INDEX=0,
            #                                  FORHAEAD_MASS=0,
            #                                  NODE_POSITION=len(self.tagdag[-1].LIST_NODE),
            #                                  NOW_AA_LIST=[2])

            self.peak_by_idx_list = self.peak_by_idx_list[1:]
            self.tagdag[-1].LIST_NODE.append(res_start_point)
            self.tagdag[-1].LIST_NODE_PEAK_INDEX.append(first_num)
            self.tagdag[-1].IN_ZERO_NODE.append(res_start_point)

            self.__get_TagDAG__(direction, res_start_point, args, model_wrapper, 0)
            if self.tagdag[-1].NUM_NODE < 5:
                self.tagdag.pop(-1)



class Tag(object):
    def __init__(self,args):

        # self.aa =  GetAAMap()

        self.args = args
        # self.graph = Graph()
        # self.feature_dp = [search_reader[serch_index]]  # 此时会执行DeepNovoDenovoDataset中的__getiem__函数
        # print(self.feature_dp)

    def search(self,model_wrapper: InferenceModelWrapper, search_reader: DeepNovoDenovoDataset,writer:TagWrite):
        log_str = ""
        for index in range(len(search_reader)):

            feature_dp = search_reader[index]
            peak_location = np.array(feature_dp.peak_location)  # [[spectrum1的moz],[spectrum2的moz],[spectrum3的moz],[spectrum4的moz]]
            peak_intensity = np.array(feature_dp.peak_intensity)
            batch_precursormass = np.array(feature_dp.precursormass)

            peak_location = torch.from_numpy(peak_location).to(device)  # 转为tensor，存储一个batch的世界谱峰质荷比
            peak_intensity = torch.from_numpy(peak_intensity).to(device)
            precursormass = torch.from_numpy(batch_precursormass).to(device)

            spectrum_state = (peak_location, peak_intensity, precursormass)

            graph = Graph(self.args,spectrum_state, feature_dp)
            graph.Create_Graph(model_wrapper,writer)
            if (index+1) % 200 == 0:
                current_rat = (float(index)/len(search_reader))*100
                log_str = "Current Progress " + str(current_rat)+"%"
                logger.info(log_str)

    # def GetGraph(self):
