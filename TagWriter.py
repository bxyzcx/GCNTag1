from dataclasses import dataclass
from data_reader import DDAFeature
import deepnovo_config
import numpy as np
import time
@dataclass
class Tag:
    direction :int
    tag_score:float
    tag_sequence :str
    Left_flanking_mass:float
    Right_flanking_mass :float
    tag_position_score : str

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

@dataclass
class TagDAGEdge:
    STAR_NODE_INDEX: int  # 此时边对应的起始节点在TagDAG中LIST_NODE中的索引是多少
    AA_LIST: list  # 此时这个边，也就是这个质量差对应的氨基酸的列表
    AA_LIST_SCORE: list  # 边对应氨基酸的打分
    END_NODE_INDEX: int  # 此时边对应的结束节点


@dataclass
class TagDAG:
    LIST_NODE: list  # 所有Node的列表
    LIST_NODE_PEAK_INDEX: list  # 所有生成的NODE对应谱峰中的哪根峰
    NUM_NODE: int   # 节点个数
    IN_ZERO_NODE:list   # 入度为0的节点列表
    BACKORFORWORD: int  # 0表示forward 1便是back

@dataclass
class total_temporary_path:
    aa_list:list  # [[],[]]
    aa_score:list
    tag_score :list  # []



@dataclass
class aa_temporary_path:
    aa_list: list
    aa_score: list
    tag_score: list

@dataclass
class node_sequece:
    aa_sequece:list
    aa_score:list


class TagWrite(object):
    def __init__(self, args,logger):
        self.args = args
        self.logger = logger
        self.output_handle2 = open(self.args.denovo_output_file + '.tagbeamsearch', 'w')
        self.Total_min_score = -5.0
        self.aa_min_score = -5.0
        self.min_len = 3
        self.precosermass = 0.0
        self.tag_min_score = -1.0  # 当前使用氨基酸阈值
        self.Top_num = args.MAX_TAG_NUM
        self.feature_id = ""
        self.tag = []
        self.forward_flag = []  # 当前节点是否被作为开始节点被遍历过 以节点在tag中的索引作为存储
        self.backward_flag = []
        # self.flag = [1 for x in range(500)]  # 创建500（谱峰个数）长的判断当前谱峰是否被遍历过
        self.flag = [1 for x in range(self.args.MAX_NUM_PEAK)]  # 创建500（谱峰个数）长的判断当前谱峰是否被遍历过
        self.start_time = 0.0  # 为了防止有遗漏回路导致代码长时间陷入死循环，当然构造的时候已经规避过这个问题，只是以防万一
        self.Tag_list = []
        self.Tag_seq_list = []
        self.Tag_score_list = []

        self.sequence_list = []
        self.score_list = []

        self.visited = []
        self.node_sequece_list = []
        self.long_sequece = []

        self.error_info_flag = True

    def close(self):
        self.output_handle2.close()


    def __write_head__(self,dda_original_feature:DDAFeature,precursor_mass):
        print('BEGIN', file=self.output_handle2, end='\n')
        feature_id = dda_original_feature.feature_id
        feature_area = dda_original_feature.feature_area
        precursor_mz = str(dda_original_feature.mz)
        self.precosermass = precursor_mass
        # print("TagWriter.py line 98 precursor_mass:",precursor_mass)
        precursor_charge = str(dda_original_feature.z)
        precursor_mass = dda_original_feature.mass
        predicted_row = "\t".join([feature_id,
                                   feature_area,
                                   precursor_mz,
                                   precursor_charge,
                                   str(precursor_mass)])

        print(predicted_row, file=self.output_handle2, end="\n")
        header_list2 = ["index",
                        "direction",
                        "tag",
                        "tag_score",
                        "tag_position_score",
                        "left_flanking_mass",
                        "right_flanking_mass"
                        ]
        header_row2 = "\t".join(header_list2)
        print(header_row2, file=self.output_handle2, end='\n')



    def __search_node__(self,tagdag:TagDAG,node:TagDAGNode,dagidx,total_score_path,aa_score_path):  # 传入dagidx为了判断是b离子还是y离子是否需要进行倒序输出

        for edge in node.LIST_EDGE:

            if not total_score_path.aa_list:
                for idx_a, aa_id in enumerate(edge.AA_LIST):
                    now_aa_socre = edge.AA_LIST_SCORE[idx_a]
                    aa_ls = [deepnovo_config.vocab_reverse[aa_id]]
                    aa_score_ls = [now_aa_socre]
                    if now_aa_socre > self.aa_min_score :
                        total_score_path.aa_list.append(aa_ls)
                        total_score_path.aa_score.append(aa_score_ls)
                        total_score_path.tag_score.append(now_aa_socre)

            else:
                for idx_a, aa_id in enumerate(edge.AA_LIST):
                    now_aa_socre = edge.AA_LIST_SCORE[idx_a]

                    if not total_score_path.aa_list:
                        now_aa_socre = edge.AA_LIST_SCORE[idx_a]
                        aa_ls = [deepnovo_config.vocab_reverse[aa_id]]
                        aa_score_ls = [now_aa_socre]
                        if now_aa_socre > self.Total_min_score:
                            total_score_path.aa_list.append(aa_ls)
                            total_score_path.aa_score.append(aa_score_ls)
                            total_score_path.tag_score = [now_aa_socre]
                            continue
                    # print("len2",len(total_score_path.aa_list),len(total_score_path.aa_score),len(total_score_path.tag_score))
                    for idx_t, total_path in enumerate(total_score_path.aa_list):
                        # print(idx_t)
                        now_total_path_score = total_score_path.tag_score[idx_t]
                        add_new_score = now_total_path_score + now_aa_socre
                        add_new_score_avg = float(add_new_score)/len(total_path)

                        if add_new_score_avg < self.Total_min_score and len(total_path)>=self.min_len:
                            tag_seq = ""
                            tag_position = ""
                            if dagidx == 0:
                                aa_list1 = total_path
                                aa_score_list1 = total_score_path.aa_score[idx_t]
                            else:
                                aa_list1 = total_path[::-1]
                                aa_score_list1 = total_score_path.aa_score[idx_t][::-1]

                            # print("aa_list1:",aa_list1)
                            for idx_aaid,aa_ in enumerate(aa_list1):
                                # print(aa_id)
                                # print(type(tag_seq),type(deepnovo_config.vocab_reverse[aa_id]))
                                tag_seq += aa_+","
                                # print(aa_score_list1[idx_aaid])
                                tag_position += str(aa_score_list1[idx_aaid])+","

                            # print("tag_seq",tag_seq)
                            tag_sc = float(now_total_path_score)/len(total_path)

                            tag = Tag(direction=dagidx+1,
                                      tag_score=tag_sc,
                                      tag_sequence=tag_seq,
                                      Left_flanking_mass=0,
                                      Right_flanking_mass=0,
                                      tag_position_score = tag_position)
                            self.Tag_list.append(tag)
                            self.Tag_score_list.append(tag_sc)
                            del(total_score_path.aa_list[idx_t])
                            del(total_score_path.aa_score[idx_t])
                            del(total_score_path.tag_score[idx_t])
                            print("newtag",tag)
                        elif add_new_score_avg < self.Total_min_score and len(total_path)<self.min_len:
                            del (total_score_path.aa_list[idx_t])
                            del (total_score_path.aa_score[idx_t])
                            del (total_score_path.tag_score[idx_t])
                        else:
                            total_score_path.aa_list[idx_t].append(deepnovo_config.vocab_reverse[aa_id])
                            total_score_path.aa_score[idx_t].append(now_aa_socre)
                            total_score_path.tag_score[idx_t] = add_new_score
            # print("tagdag",type(tagdag))
            next_node =tagdag.LIST_NODE[edge.END_NODE_INDEX]
            self.__search_node__(tagdag,next_node,dagidx,total_score_path,aa_score_path)


        for idx_tp,total_path in enumerate(total_score_path.aa_list):
            if len(total_path) >= self.min_len:
                tag_sc = float(total_score_path.tag_score[idx_tp]) / len(total_path)
                tag_seq = ""
                tag_position = ""
                if dagidx == 0:
                    aa_list1 = total_path
                    aa_score_list1 = total_score_path.aa_score[idx_tp]
                else:
                    aa_list1 = total_path[::-1]
                    aa_score_list1 = total_score_path.aa_score[idx_tp][::-1]
                tag = Tag(direction=dagidx,
                          tag_score=tag_sc,
                          tag_sequence=tag_seq,
                          Left_flanking_mass=0.0,
                          Right_flanking_mass=0.0,
                          tag_position_score=tag_position)

                self.Tag_list.append(tag)
                self.Tag_score_list.append(tag_sc)
                del (total_score_path.aa_list[idx_tp])
                del (total_score_path.aa_score[idx_tp])
                del (total_score_path.tag_score[idx_tp])

        node.LIST_EDGE = []


    def __extag__(self,tagdag:TagDAG,node:TagDAGNode,sequence_list:list,score_list:list,tag_sumsccore_list:list,start_mass,tag_len_list:list):
        if time.time() - self.start_time > 30:
            if self.error_info_flag == True:
                abnormal_str = " abnormal spectrum,timeout" + self.feature_id
                # self.logger.info(abnormal_str)
                self.error_info_flag = False
            return
        # append_flag = []
        for edi, edge in enumerate(node.LIST_EDGE):
            ls = []
            ls_score = []
            ls_sum = []
            ls_taglen = []
            now_start_mass = start_mass
            # if len(sequence_list) == 1 and sequence_list[0] == "" and self.flag[node.MOZ_INDEX] == 0:
            #     continue
            # print("line 245: len of test",len(self.test))
            if len(sequence_list) == 1 and sequence_list[0] == "" and self.forward_flag[node.NODE_POSITION]==0 and tagdag.BACKORFORWORD==0:
                continue
            elif len(sequence_list) == 1 and sequence_list[0] == "" and self.backward_flag[node.NODE_POSITION]==0 and tagdag.BACKORFORWORD== 1:
                continue

             # 如果sequence_list为[]空里面没有 则进行深度优先遍历直到遍历到一个大于-3.5的节点则作为序列标签的开始
            if not sequence_list:
                for idx_a, aa_id in enumerate(edge.AA_LIST):
                    edge_aa_score = str(edge.AA_LIST_SCORE[idx_a])
                    aa_ls = str(deepnovo_config.vocab_reverse[aa_id])
                    #最好的是应该找到第一个大于阈值的点开始作为起始节点
                    # if float(edge.AA_LIST_SCORE[idx_a]) > -3.5:
                    if float(edge.AA_LIST_SCORE[idx_a]) >self.tag_min_score:
                        # sequence_list.append(aa_ls)
                        # score_list.append(edge_aa_score)
                        # tag_sumsccore_list.append(float(edge.AA_LIST_SCORE[idx_a]))
                        # tag_len_list.append(1)
                        now_start_mass = node.MASS
                        ls.append(aa_ls)
                        ls_score.append(edge_aa_score)
                        ls_sum.append(float(edge.AA_LIST_SCORE[idx_a]))
                        ls_taglen.append(1)
                        # append_flag.append(1)
                # print("sequence_list",sequence_list)

            else:  # 如果sequence——list已经存储了氨基酸
                for idx_aa_id, aa_id in enumerate(edge.AA_LIST):
                    for idx, seq in enumerate(sequence_list):  # 当前tag的sequence逐一与连边的氨基酸可能的打饭进行判断是否大于-1.0

                        edge_aalist_AA_score = float(edge.AA_LIST_SCORE[idx_aa_id])
                        add_aa_avg_score = float(tag_sumsccore_list[idx] + edge_aalist_AA_score) / (tag_len_list[idx]+1)
                        # if node.NODE_POSITION == 308:
                        #     print("now id",aa_id)

                        if add_aa_avg_score > self.tag_min_score:  # 如果当前的sequence与连边的氨基酸拼接后的得分大于-1.0


                            if seq == "":  # 如果是刚清空了sequencelist的状态即[""] 则不需要“，”
                                ss = str(deepnovo_config.vocab_reverse[aa_id])
                                ss_score = str(edge.AA_LIST_SCORE[idx_aa_id])
                                now_sum = float(ss_score)
                                tag_l = 1
                            else:
                                ss = seq + "," + str(deepnovo_config.vocab_reverse[aa_id])
                                ss_score = score_list[idx] + "," + str(edge.AA_LIST_SCORE[idx_aa_id])
                                now_sum = tag_sumsccore_list[idx] + float(edge.AA_LIST_SCORE[idx_aa_id])
                                tag_l = tag_len_list[idx]+1
                            # 因为当前序列再加上当前遍历的边的氨基酸的打分是大于-1.0的所以继续存储并且送入更深的地方
                            ls.append(ss)
                            ls_score.append(ss_score)
                            ls_sum.append(now_sum)
                            ls_taglen.append(tag_l)


                        elif add_aa_avg_score <= self.tag_min_score and tag_len_list[idx]<3:  #如果当前序列加上当前遍历的边的氨基酸后序列小于-1.0但是长度也不够3那么清除当前序列不会继续向下传递
                            if "" not  in ls:  # 如果已经有了""在列表里则不在重复添加以免后续遍历
                                ls.append("")
                                ls_score.append("")
                                ls_sum.append(0.0)
                                ls_taglen.append(0)
                                now_start_mass = tagdag.LIST_NODE[edge.END_NODE_INDEX].MASS


                        elif add_aa_avg_score <= self.tag_min_score and tag_len_list[idx]>=3:  #如果当前序列加上当前遍历的边的氨基酸得分小于-1.0但是长度大于3我们需要将其存储
                            tag_score = float(tag_sumsccore_list[idx]) / tag_len_list[idx]
                            now_start_mass = tagdag.LIST_NODE[edge.END_NODE_INDEX].MASS
                            if seq not in self.Tag_seq_list:
                                now_seq = seq
                                # print("seq not in tag_sequece_list", seq not in self.Tag_seq_list)
                                if tagdag.BACKORFORWORD == 1:
                                    seq = self.__backward_seq__(seq)
                                tag = Tag(direction=tagdag.BACKORFORWORD,
                                          tag_score=tag_score,
                                          tag_sequence=seq,
                                          Left_flanking_mass=start_mass,
                                          Right_flanking_mass=self.precosermass - node.MASS,
                                          tag_position_score=score_list[idx])
                                # print("1......tag",tag)
                                # print("seq in tag_sequece_list", seq in self.Tag_seq_list)


                                self.Tag_list.append(tag)
                                self.Tag_seq_list.append(now_seq)
                                self.Tag_score_list.append(tag_score)
                            else:
                                tag_index = self.Tag_seq_list.index(seq)
                                score = max(self.Tag_score_list[tag_index],tag_score)
                                self.Tag_score_list[tag_index] = score
                                self.Tag_list[tag_index].tag_score = score

                            if "" not in ls:
                                ls.append("")
                                ls_score.append("")
                                ls_sum.append(0.0)
                                ls_taglen.append(0)


                        # if node.NODE_POSITION == 318:
                        #     # print("154...edge_aalist_AA_score",edge_aalist_AA_score)
                        #     print(".............................................")
                        #     print("edge_aa,aa_score", aa_id, edge_aalist_AA_score)
                        #     print("add_aa_avg_score",add_aa_avg_score)
                        #     print("nowseq",ls)
                        #     print("nowscore",ls)
                        #     print("edge.AA_LIST,edge.AA_LIST_SCORE",edge.AA_LIST,edge.AA_LIST_SCORE)
                        #     print("node.LIST_EDGE",node.LIST_EDGE)

                if len(ls) == 0:
                    ls.append("")
                    ls_score.append("")
                    ls_sum.append(0.0)
                    ls_taglen.append(0)
                # 修改INNAII提取不到问题
                # sequence_list = ls
                # score_list = ls_score
                # tag_sumsccore_list = ls_sum
                # tag_len_list = ls_taglen

            # if node.NODE_POSITION == 0:
            #     # print("154...edge_aalist_AA_score",edge_aalist_AA_score)
            #     # print("154...add_aa_avg_score", add_aa_avg_score)
            #     print("nowseq", ls)
            #     print("nowscore", ls_score)
            #     print("edge.AA_LIST,edge.AA_LIST_SCORE", edge.AA_LIST, edge.AA_LIST_SCORE)
            #     print("node.LIST_EDGE", node.LIST_EDGE)
            #     # print("edge_aa,aa_score", aa_id, edge_aalist_AA_score)
            next_node = tagdag.LIST_NODE[edge.END_NODE_INDEX]
            # 修改INNAII提取不到问题
            # self.__extag__(tagdag, next_node, sequence_list, score_list,tag_sumsccore_list,start_mass,tag_len_list)
            self.__extag__(tagdag, next_node, ls, ls_score,ls_sum,now_start_mass,ls_taglen)

            # print("len score_list,sequecen_list", len(score_list), len(sequence_list),score_list,sequence_list)
        # if len(sequence_list) == 1 and sequence_list[0] == "":
        #     self.flag[node.MOZ_INDEX] =0

        if tagdag.BACKORFORWORD == 0 and len(sequence_list) == 1 and sequence_list[0] == "":
            self.forward_flag[node.NODE_POSITION] = 0
        elif tagdag.BACKORFORWORD == 1 and len(sequence_list) == 1 and sequence_list[0] == "":
            self.backward_flag[node.NODE_POSITION] = 0

        if node.OUT == 0:
            for seq_index, sequece_i in enumerate(sequence_list):
                if len(sequece_i) >= 4:
                    tag_score = float(tag_sumsccore_list[seq_index]) / tag_len_list[seq_index]

                    if sequece_i not in self.Tag_seq_list:
                        now_sequence_i = sequece_i
                        # print("sequece_i not in self.Tag_seq_list",sequece_i not in self.Tag_seq_list)
                        if tagdag.BACKORFORWORD == 1:
                            sequece_i = self.__backward_seq__(sequece_i)
                        tag = Tag(direction=tagdag.BACKORFORWORD,
                                  tag_score=tag_score,
                                  tag_sequence=sequece_i,
                                  Left_flanking_mass=start_mass,
                                  Right_flanking_mass=self.precosermass - node.MASS,
                                  tag_position_score=score_list[seq_index])
                        # print("2......tag", tag)
                        self.Tag_list.append(tag)
                        self.Tag_seq_list.append(now_sequence_i)
                        self.Tag_score_list.append(tag_score)
                    else:
                        tag_index = self.Tag_seq_list.index(sequece_i)
                        score = max(self.Tag_score_list[tag_index], tag_score)
                        self.Tag_score_list[tag_index] = score
                        self.Tag_list[tag_index].tag_score = score

            # print("sequence_list",sequence_list)

    @staticmethod
    def __backward_seq__(seq):
        # print("TagWriter py line 459",seq)
        if "(" in seq:
            start_point = 0
            ls_seq = seq.split(",")
            seq_new = ""
            for i in range(len(ls_seq)):
                seq_new += ls_seq[len(ls_seq) - 1 - i] + ","
            # print(seq_new)
            # print(seq)
            seq_new = seq_new[:-1]
        else:
            seq_new = seq[::-1]
        return seq_new

    # print("end")
    #包含C(+57修饰)
    def __extagincludeC__(self, tagdag: TagDAG, node: TagDAGNode, sequence_list: list, score_list: list,
                  tag_sumsccore_list: list, start_mass, tag_len_list: list):
        # append_flag = []
        for edi, edge in enumerate(node.LIST_EDGE):
            ls = []
            ls_score = []
            ls_sum = []
            ls_taglen = []
            now_start_mass = start_mass
            # 如果sequence_list为[]空里面没有 则进行深度优先遍历直到遍历到一个大于-3.5的节点则作为序列标签的开始
            if len(sequence_list) == 1 and sequence_list[0] == "" and self.flag[node.MOZ_INDEX] == 0:
                continue
            # if len(sequence_list) == 1 and sequence_list[0] == "" and self.forward_flag[node.NODE_POSITION]==0 and tagdag.BACKORFORWORD==0:
            #     continue
            # elif len(sequence_list) == 1 and sequence_list[0] == "" and self.backward_flag[node.NODE_POSITION]==0 and tagdag.BACKORFORWORD== 1:
            #     continue

            if not sequence_list:
                for idx_a, aa_id in enumerate(edge.AA_LIST):
                    edge_aa_score = str(edge.AA_LIST_SCORE[idx_a])
                    aa_ls = str(deepnovo_config.vocab_reverse[aa_id])
                    if float(edge.AA_LIST_SCORE[idx_a]) > -3.5 :
                        # sequence_list.append(aa_ls)
                        # score_list.append(edge_aa_score)
                        # tag_sumsccore_list.append(float(edge.AA_LIST_SCORE[idx_a]))
                        # tag_len_list.append(1)
                        now_start_mass = node.MASS
                        ls.append(aa_ls)
                        ls_score.append(edge_aa_score)
                        ls_sum.append(float(edge.AA_LIST_SCORE[idx_a]))
                        ls_taglen.append(1)

                        # append_flag.append(1)
                # print("sequence_list",sequence_list)

            else:  # 如果sequence——list已经存储了氨基酸

                for idx_aa_id, aa_id in enumerate(edge.AA_LIST):
                    for idx, seq in enumerate(sequence_list):  # 当前tag的sequence逐一与连边的氨基酸可能的打饭进行判断是否大于-1.0

                        edge_aalist_AA_score = float(edge.AA_LIST_SCORE[idx_aa_id])
                        add_aa_avg_score = float(tag_sumsccore_list[idx] + edge_aalist_AA_score) / (
                                    tag_len_list[idx] + 1)
                        # if node.NODE_POSITION == 308:
                        #     print("now id",aa_id)
                        if add_aa_avg_score > self.tag_min_score:  # 如果当前的sequence与连边的氨基酸拼接后的得分大于-1.0

                            if seq == "":  # 如果是刚清空了sequencelist的状态即[""] 则不需要“，”
                                ss = str(deepnovo_config.vocab_reverse[aa_id])
                                ss_score = str(edge.AA_LIST_SCORE[idx_aa_id])
                                now_sum = float(ss_score)
                                tag_l = 1
                            else:
                                ss = seq + "," + str(deepnovo_config.vocab_reverse[aa_id])
                                ss_score = score_list[idx] + "," + str(edge.AA_LIST_SCORE[idx_aa_id])
                                now_sum = tag_sumsccore_list[idx] + float(edge.AA_LIST_SCORE[idx_aa_id])
                                tag_l = tag_len_list[idx] + 1
                            # 因为当前序列再加上当前遍历的边的氨基酸的打分是大于-1.0的所以继续存储并且送入更深的地方
                            ls.append(ss)
                            ls_score.append(ss_score)
                            ls_sum.append(now_sum)
                            ls_taglen.append(tag_l)


                        elif add_aa_avg_score <= self.tag_min_score and tag_len_list[idx] < 3:  # 如果当前序列加上当前遍历的边的氨基酸后序列小于-1.0但是长度也不够3那么清除当前序列不会继续向下传递
                            if "" not in ls:  # 如果已经有了""在列表里则不在重复添加以免后续遍历
                                ls.append("")
                                ls_score.append("")
                                ls_sum.append(0.0)
                                ls_taglen.append(0)
                                now_start_mass = tagdag.LIST_NODE[edge.END_NODE_INDEX].MASS

                                # if tagdag.BACKORFORWORD == 0:
                                #     self.forward_flag[edge.END_NODE_INDEX] == 0
                                # else:
                                #     self.backward_flag[edge.END_NODE_INDEX] == 0

                        elif add_aa_avg_score <= self.tag_min_score and tag_len_list[
                            idx] >= 3:  # 如果当前序列加上当前遍历的边的氨基酸得分小于-1.0但是长度大于3我们需要将其存储
                            tag_score = float(tag_sumsccore_list[idx]) / tag_len_list[idx]
                            now_start_mass = tagdag.LIST_NODE[edge.END_NODE_INDEX].MASS
                            if seq not in self.Tag_seq_list:
                                now_seq = seq
                                # print("seq not in tag_sequece_list", seq not in self.Tag_seq_list)
                                if tagdag.BACKORFORWORD == 1:
                                    seq = self.__backward_seq__(seq)
                                tag = Tag(direction=tagdag.BACKORFORWORD,
                                          tag_score=tag_score,
                                          tag_sequence=seq,
                                          Left_flanking_mass=start_mass,
                                          Right_flanking_mass=self.precosermass - node.MASS,
                                          tag_position_score=score_list[idx])
                                # print("1......tag",tag)
                                # print("seq in tag_sequece_list", seq in self.Tag_seq_list)
                                self.Tag_list.append(tag)
                                self.Tag_seq_list.append(now_seq)

                                self.Tag_score_list.append(tag_score)
                            else:
                                tag_index = self.Tag_seq_list.index(seq)
                                score = max(self.Tag_score_list[tag_index], tag_score)
                                self.Tag_score_list[tag_index] = score
                                self.Tag_list[tag_index].tag_score = score


                            if "" not in ls:
                                ls.append("")
                                ls_score.append("")
                                ls_sum.append(0.0)
                                ls_taglen.append(0)


                        # if node.NODE_POSITION == 318:
                        #     # print("154...edge_aalist_AA_score",edge_aalist_AA_score)
                        #     print(".............................................")
                        #     print("edge_aa,aa_score", aa_id, edge_aalist_AA_score)
                        #     print("add_aa_avg_score",add_aa_avg_score)
                        #     print("nowseq",ls)
                        #     print("nowscore",ls)
                        #     print("edge.AA_LIST,edge.AA_LIST_SCORE",edge.AA_LIST,edge.AA_LIST_SCORE)
                        #     print("node.LIST_EDGE",node.LIST_EDGE)

                if len(ls) == 0:
                    ls.append("")
                    ls_score.append("")
                    ls_sum.append(0.0)
                    ls_taglen.append(0)

            # if node.NODE_POSITION == 0:
            #     # print("154...edge_aalist_AA_score",edge_aalist_AA_score)
            #     # print("154...add_aa_avg_score", add_aa_avg_score)
            #     print("nowseq", ls)
            #     print("nowscore", ls_score)
            #     print("edge.AA_LIST,edge.AA_LIST_SCORE", edge.AA_LIST, edge.AA_LIST_SCORE)
            #     print("node.LIST_EDGE", node.LIST_EDGE)
            #     # print("edge_aa,aa_score", aa_id, edge_aalist_AA_score)
            next_node = tagdag.LIST_NODE[edge.END_NODE_INDEX]
            # 修改INNAII提取不到问题
            # self.__extag__(tagdag, next_node, sequence_list, score_list,tag_sumsccore_list,start_mass,tag_len_list)
            self.__extagincludeC__(tagdag, next_node, ls, ls_score, ls_sum, now_start_mass, ls_taglen)

            # print("len score_list,sequecen_list", len(score_list), len(sequence_list),score_list,sequence_list)
        if len(sequence_list) == 1 and sequence_list[0] == "":
            self.flag[node.MOZ_INDEX] = 0
        # if tagdag.BACKORFORWORD == 0 and len(sequence_list) == 1 and sequence_list[0] == "":
        #     self.forward_flag[node.NODE_POSITION] = 0
        # elif tagdag.BACKORFORWORD == 1 and len(sequence_list) == 1 and sequence_list[0] == "" :
        #     self.backward_flag[node.NODE_POSITION] = 0

        if node.OUT == 0:

            for seq_index, sequece_i in enumerate(sequence_list):
                if len(sequece_i) >= 4:
                    tag_score = float(tag_sumsccore_list[seq_index]) / tag_len_list[seq_index]

                    if sequece_i not in self.Tag_seq_list:
                        now_sequence_i = sequece_i
                        # print("sequece_i not in self.Tag_seq_list",sequece_i not in self.Tag_seq_list)
                        if tagdag.BACKORFORWORD == 1:
                            # sequece_i = sequece_i[::-1]
                            sequece_i = self.__backward_seq__(sequece_i)
                        tag = Tag(direction=tagdag.BACKORFORWORD,
                                  tag_score=tag_score,
                                  tag_sequence=sequece_i,
                                  Left_flanking_mass=start_mass,
                                  Right_flanking_mass=self.precosermass - node.MASS,
                                  tag_position_score=score_list[seq_index])
                        # print("2......tag", tag)
                        self.Tag_list.append(tag)
                        self.Tag_seq_list.append(now_sequence_i)
                        self.Tag_score_list.append(tag_score)
                    else:
                        tag_index = self.Tag_seq_list.index(sequece_i)
                        score = max(self.Tag_score_list[tag_index], tag_score)
                        self.Tag_score_list[tag_index] = score
                        self.Tag_list[tag_index].tag_score = score

            # print("sequence_list",sequence_list)

    def __from_longsequece_ex_tag__(self,backorfor):
        tag_ls = []
        score_ls = []

        for seq_idx,seq in enumerate(self.sequence_list):
            # if seq.startswith("Y"):
            #     if not seq.startswith("Y,T"):
            #         print("seq",seq)
            # if seq_idx % 10000==0:
            #     print("now sequece/total sequece",seq_idx,",",len(self.sequence_list))

            seq_ls = seq.split(",")
            sls = self.score_list[seq_idx].split(",")
            sc_ls = []
            sequnce_sum = 0.0
            for x in sls:
                f_x = float(x)
                sequnce_sum += f_x
                sc_ls.append(f_x)

            seq_avg = sequnce_sum / len(sc_ls)
            tag = ""
            tag_num = 0
            tag_sum = 0.0
            tag_score_str = ""
            tag_avg  = 0.0
            if seq_avg > -1.0:
                if seq not in tag_ls:
                    tag_ls.append(tag)
                    score_ls.append(seq_avg)
                    continue

            for aa_idx,aainseq in enumerate(seq_ls):
                # print("tag:", tag,sc_ls[aa_idx])
                if tag == "" and sc_ls[aa_idx] > -3.0:
                    tag = aainseq+","
                    tag_score_str = str(sc_ls[aa_idx]) + ","
                    tag_num += 1
                    tag_sum += sc_ls[aa_idx]

                elif tag != "":

                    now_score_sum = tag_sum + sc_ls[aa_idx]
                    now_avg = now_score_sum / (tag_num+1)
                    # if tag == "Y,I":
                    #     print("Y,I",now_avg)
                    # 如果tag_num 再添加一个氨基酸就会<-1.0并且长度大于等于三

                    if now_avg < -1.0 and tag_num >= 3 and (tag not in tag_ls):
                        # print("tag:",tag)
                        tag_ls.append(tag)
                        score_ls.append(tag_sum / tag_num)

                        tag = ""
                        tag_num = 0
                        tag_sum = 0.0
                        tag_score_str = ""
                    elif now_avg < -1.0 and tag_num < 3:
                        tag = ""
                        tag_num = 0
                        tag_sum = 0.0
                        tag_score_str = ""

                    elif now_avg > -1.0:
                        tag = tag + aainseq+","
                        tag_score_str = tag_score_str + str(sc_ls[aa_idx]) + ","
                        tag_num += 1
                        tag_sum += float(sc_ls[aa_idx])

            if tag_num >= 3 and (tag not in tag_ls):
                tag_ls.append(tag)
                # print("tag:", tag)
                score_ls.append(tag_sum / tag_num)
                tag = ""
                tag_num = 0
                tag_sum = 0.0
                tag_score_str = ""

        tag_score_list_np = np.array(score_ls)
        index_list = np.argsort(tag_score_list_np)[::-1]
        print("len:",len(index_list))
        for idx in index_list:
            print("tag,score:",tag_ls[idx],score_ls[idx])


    def __shendu__(self,tagdag:TagDAG,node:TagDAGNode,sequence_list,score_list):
        # print("mass", type(node.MASS),node.MASS)
        if str(node.MASS).startswith("891"):
            print("891..............................:",node.NODE_POSITION,node.MASS)
            print("node.LIST_EDGE",node.LIST_EDGE)
            for e in node.LIST_EDGE:
                print("1......",tagdag.LIST_NODE[e.END_NODE_INDEX].MASS)
        for edi, edge in enumerate(node.LIST_EDGE):
            if not sequence_list:
                for idx_a, aa_id in enumerate(edge.AA_LIST):
                    edge_aa_score = str(edge.AA_LIST_SCORE[idx_a])
                    aa_ls = str(deepnovo_config.vocab_reverse[aa_id])

                    sequence_list.append(aa_ls)
                    score_list.append(edge_aa_score)
                # print("sequence_list",sequence_list)
            else:
                ls = []
                ls_score = []
                for idx_a, aa_id in enumerate(edge.AA_LIST):
                    for idx, seq in enumerate(sequence_list):
                        if seq == "":
                            ss = str(deepnovo_config.vocab_reverse[aa_id])
                            ss_score = str(edge.AA_LIST_SCORE[idx_a])
                        else:
                            ss = seq+"," +str(deepnovo_config.vocab_reverse[aa_id])
                            ss_score = score_list[idx] + "," + str(edge.AA_LIST_SCORE[idx_a])
                        if ss == 'K,G,P,T,D,T,A,G,V,P,I,T,D,T,N,N,A,Q':
                            print("line483:",node.NODE_POSITION,tagdag.LIST_NODE[edge.END_NODE_INDEX].MASS,node.MASS)
                        ls.append(ss)
                        ls_score.append(ss_score)

                sequence_list = ls
                score_list = ls_score

                # if node.NODE_POSITION == 57:
                # if node.MASS == 1016.60888671875:
                #     print("line400",node.MASS)
                #     print("node.index",node.NODE_POSITION)
                #     print("node的候选离子",node.LIST_EDGE)
                #     print("node edge_aa_list",edge.AA_LIST)
                #     print("node edge_aa_list",edge.AA_LIST_SCORE)
                #     print("edge.END_NODE_INDEX", edge.END_NODE_INDEX)
                    # print("sequence_list", sequence_list)
                    # print("score_list", score_list)
                #     print("sequece_list",sequence_list)

            next_node = tagdag.LIST_NODE[edge.END_NODE_INDEX]
            self.__shendu__(tagdag, next_node, sequence_list, score_list)
            for idx, seq in enumerate(sequence_list):
                if "," not in seq:
                    re_score_list = [""]
                    re_seq_list = [""]
                    # sequence_list[idx] = ""
                    # score_list[idx] = ""

                    sequence_list = re_seq_list
                    score_list[idx] = re_score_list
                    break

                else:
                    s_score = score_list[idx]
                    s = seq

                    bi = s.rindex(",")
                    bi_score = s_score.rindex(",")

                    ss = s[:bi]
                    ss_score = s_score[:bi_score]

                    sequence_list[idx] = ss
                    score_list[idx] = ss_score

        # if node.NODE_POSITION == 0:
        #     print("146 node.OUT:", node.OUT)
        #     print("sequence_list", sequence_list)
        #     print("score_list", score_list)

        if node.OUT == 0:
            for seq_index, sequece_i in enumerate(sequence_list):
                if len(sequece_i) >= 5:
                    # if sequece_i.startswith("Y"):
                    #     if not sequece_i.startswith("Y,T"):
                    #         print("sequece_i", sequece_i,score_list[seq_index])
                    self.sequence_list.append(sequece_i)
                    self.score_list.append(score_list[seq_index])
            # print("sequence_list",sequence_list,score_list)

        # print("end")


    def __shendu2__(self,tagdag:TagDAG,node:TagDAGNode,sequence_list,score_list):
        back_sequece_list_return = []
        back_score_list_return = []
        if self.visited[node.NODE_POSITION] == 0 and node.OUT != 0:
            for seq_idx,sequece_inls in enumerate(sequence_list):
                # print(self.node_sequece_list)
                for node_se_idx,node_se in enumerate(self.node_sequece_list[node.NODE_POSITION].aa_sequece):
                    now_sueqence = sequece_inls+ "," + node_se
                    # print("score_list",type(score_list))
                    # print("self.node_sequece_list[node.NODE_POSITION].aa_score",self.node_sequece_list[node.NODE_POSITION].aa_score)
                    now_sueqence_score = score_list[seq_idx] + ","+ self.node_sequece_list[node.NODE_POSITION].aa_score[node_se_idx]
                    print("now_sueqence",now_sueqence)
                    self.sequence_list.append(now_sueqence)
                    self.score_list.append(now_sueqence_score)

            # 返回backsequence，此时的backsequence为此节点的上一个氨基酸以及此节点保存的氨基酸
            back_sequece_list = []
            back_sequece_score_list = []
            bk_ls = []
            bk_score_ls = []
            for idx, seq in enumerate(sequence_list):
                s = seq
                bi = s.rindex(",")
                ss = s[bi + 1:]

                s_sc = score_list[idx]
                bi_sc = s_sc.rindex(",")
                score_strs = s_sc[:bi_sc]

                if ss not in back_sequece_list:
                    back_sequece_list.append(ss)  # 前面不包含逗号“,”
                    back_sequece_score_list.append(score_strs)
                    for node_seq_idx, node_seq in enumerate(self.node_sequece_list[node.NODE_POSITION].aa_sequece):
                        seq = ss+","+ node_seq
                        sco = score_strs + ","+ self.node_sequece_list[node.NODE_POSITION].aa_score[node_seq_idx]
                        bk_ls.append(seq)
                        bk_score_ls.append(sco)
            return bk_ls,bk_score_ls

        # 从此处开始运行每个节点
        for edi, edge in enumerate(node.LIST_EDGE):
            if not sequence_list:
                for idx_a, aa_id in enumerate(edge.AA_LIST):
                    aa_ls = str(deepnovo_config.vocab_reverse[aa_id])
                    sequence_list.append(aa_ls)
                    score_list.append(edge.AA_LIST_SCORE)
            else:
                ls = []
                s_ls = []
                for idx_a, aa_id in enumerate(edge.AA_LIST):
                    for idx, seq in enumerate(sequence_list):
                        ss = seq+"," + str(deepnovo_config.vocab_reverse[aa_id])
                        now_str_score = str(score_list[idx])
                        if "[" in now_str_score:
                            score_s = now_str_score[1:-1] +","+str(edge.AA_LIST_SCORE[idx_a])
                        else:
                            score_s = now_str_score + "," + str(edge.AA_LIST_SCORE[idx_a])

                        ls.append(ss)
                        s_ls.append(score_s)

                sequence_list = ls
                score_list = s_ls
                    # print("sequece_list",sequence_list)/
            next_node = tagdag.LIST_NODE[edge.END_NODE_INDEX]
            # print("seuqunce:1",sequence_list)
            back_seq_ls,back_score_ls = self.__shendu2__(tagdag, next_node,sequence_list,score_list)
            # print("seuqunce:2", sequence_list)
            for back_seq_idx, back_seq in enumerate(back_seq_ls):
                back_sequece_list_return.append(back_seq)
                back_score_list_return.append(back_score_ls[back_seq_idx])
            # back_sequece_list_return.append(back_seq_ls)
            # back_score_list_return.append(back_score_ls)

            for idx, seq in enumerate(sequence_list):
                s = seq
                sc = score_list[idx]
                try:
                    bi = s.rindex(",")
                    ss = s[:bi]

                    bi_score = sc.rindex(",")
                    score_strsequ = sc[:bi_score]
                except:
                    ss  = s
                    score_strsequ = sc
                    print("total end")

                sequence_list[idx] = ss
                score_list[idx] = score_strsequ
        # 结束循环后的操作
        # print("sequence_list",sequence_list)
        back_sequece_list = []
        back_sequece_score_list = []
        # print("sequence_list:",sequence_list)
        for idx, seq in enumerate(sequence_list):
            s = seq
            sc = score_list[idx]
            # try:
            # print("s",s)
            bi = s.rindex(",")
            bi_score = sc.rindex(",")

            ss = s[bi+1:]
            # print("score_strs:",score_strs,type(score_strs),len(score_strs))
            score_strsequ = sc[bi_score+1:]
            # except:
            #     print("next node")
                # ss = s
                # score_strsequ = sc

            if ss not in back_sequece_list:
                back_sequece_list.append(ss)  # 前面不包含逗号“,”
                back_sequece_score_list.append(score_strsequ)

        self.visited[node.NODE_POSITION] = 0
        if node.OUT == 0:
            for s_idx,se in enumerate(sequence_list):
                self.sequence_list.append(se)
                self.score_list.append(score_list[s_idx])
                print("se:",se)
                # print("back_sequece_list,back_sequece_score_list:",back_sequece_list,back_sequece_score_list)
            return back_sequece_list, back_sequece_score_list

        else:
            bk_ls = []
            bk_scroe_ls = []
            for idx_i,seq_back_ls in enumerate(back_sequece_list):
                for idx_rei,re_seq in enumerate(back_sequece_list_return):
                    # for idx_rei_seq,ret_ls_seq in ret_ls:
                    n_back_seq = seq_back_ls +"," + re_seq
                    n_back_score = back_sequece_score_list[idx_i] +","+back_score_list_return[idx_rei]
                    bk_ls.append(n_back_seq)
                    bk_scroe_ls.append(n_back_score)

            if node.IN > 1:
                n_s = node_sequece(aa_sequece=back_sequece_list_return,aa_score=back_score_list_return)
                self.node_sequece_list[node.NODE_POSITION] = n_s


            # print("end")
            # print("now",)
            return bk_ls, bk_scroe_ls


    def __get_Tag__(self, tagdag_list,feature_id):
        # print("977 start get_tag")
        self.Tag_list = []
        self.Tag_score_list = []
        self.tag_sequnce_list = []
        # self.forward_flag = [1 for x in range(500)]
        # self.backward_flag = [1 for x in range(500)]
        self.feature_id = feature_id
        self.error_info_flag = True
        self.forward_flag = [1 for x in range(self.args.MAX_NUM_PEAK)]
        self.backward_flag = [1 for x in range(self.args.MAX_NUM_PEAK)]

        for dagidx,tagdag in enumerate(tagdag_list):
            if tagdag.NUM_NODE < 5:
                continue
            bianli_tagdag = 0
            befor_MASS = 0.0
            for nod_idx,zero_in_node in enumerate(tagdag.IN_ZERO_NODE):
                # print(len(tagdag_list),dagidx)
                # print(nod_idx)

                if zero_in_node.MASS == befor_MASS:
                    continue
                befor_MASS = zero_in_node.MASS
                self.start_time = time.time()
                # total_score_path = total_temporary_path(aa_list=[],aa_score=[],tag_score=[])
                # aa_score_path = total_temporary_path(aa_list=[],aa_score=[],tag_score=[])
                # self.__search_node__(tagdag, zero_in_node, dagidx, total_score_path, aa_score_path)
                # self.sequence_list = []
                self.visited = [1 for x in range(len(tagdag.LIST_NODE))]
                self.sequence_list = []
                self.score_list = []

                start_mass = zero_in_node.MASS
                self.__extag__(tagdag,zero_in_node, [], [], [], start_mass, [])

                bianli_tagdag = bianli_tagdag + 1

        # print("taglist",self.Tag_list)


    def __nowinit__(self):
        self.tag = []
        self.forward_flag = []  # 当前节点是否被作为开始节点被遍历过 以节点在tag中的索引作为存储
        self.backward_flag = []

        self.Tag_list = []
        self.Tag_seq_list = []
        self.Tag_score_list = []

        self.sequence_list = []
        self.score_list = []

        self.visited = []
        self.node_sequece_list = []
        self.long_sequece = []


    def __write_Tag__(self,dda_original_feature: DDAFeature, tagdag_list,precursor_mass):
        # print("TagWriter.py 1034")
        self.__nowinit__()
        self.__write_head__(dda_original_feature ,precursor_mass)
        tag_list = []
        feature_id = dda_original_feature.feature_id
        self.__get_Tag__(tagdag_list,feature_id)
        # print("length of tag list ",len(self.Tag_list))
        self.__write_File()
        # print("END tag write")


    def __write_File(self):
        tag_score_list_np = np.array(self.Tag_score_list)
        index_list = np.argsort(tag_score_list_np)[::-1]
        for id, listidx in enumerate(index_list):
            # print(id,"id")
            # print(Tag_list[listidx])
            # print("tag:",Tag_list[listidx].tag_sequence)
            if id >= self.Top_num:
                break
            predicted_row = "\t".join([str(id),
                                       str(self.Tag_list[listidx].direction),
                                       self.Tag_list[listidx].tag_sequence,
                                       str(self.Tag_list[listidx].tag_score),
                                       self.Tag_list[listidx].tag_position_score,
                                       str(self.Tag_list[listidx].Left_flanking_mass),
                                       str(self.Tag_list[listidx].Right_flanking_mass)])
            print(predicted_row, file=self.output_handle2, end="\n")
        print('END', file=self.output_handle2, end="\n")


