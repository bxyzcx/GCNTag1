# Copyright 2019 Rui Qiao. All Rights Reserved.
#
# DeepNovoV2 is publicly available for non-commercial uses.
# ==============================================================================
import time
import torch
import logging
import logging.config
import os
from train_func1 import build_model

from data_reader import DeepNovoDenovoDataset
from model_gcn import InferenceModelWrapper
from init_args import init_args
from SpectralGraph import Tag
from TagWriter import TagWrite
import datetime
logger = logging.getLogger(__name__)



def run_CGNTag(args):
    # start!
    torch.cuda.empty_cache()
    start = time.time()
    logger.info("denovo mode")
    data_reader = DeepNovoDenovoDataset(feature_filename=args.denovo_input_feature_file,
                                        spectrum_filename=args.denovo_input_spectrum_file,
                                        args=args)
    forward_deepnovo, backward_deepnovo, init_net = build_model(args=args, training=False)
    model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
    writer = TagWrite(args,logger)

    tag = Tag(args=args)
    tag.search(model_wrapper, data_reader, writer)

    print('using time:', time.time() - start)



def init_log(log_file_name):
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)


if __name__ == '__main__':
    param_path = "./param/cross.9high_80k.exclude_bacillus_4_PeakNetwork_NH3H2O_InternalIons_Edge.cfg"
    log_path = "./log/AblationStudy/[test77]"
    if os.path.isfile(param_path):
        log_path += (param_path.split("/")[-1] + "_")
        dir, param_file = os.path.split(param_path)
        # log_file_name = "top5_" + param_file[-4] + ".log"
        now = datetime.datetime.now().strftime("%Y%m%d%H%M")
        args = init_args(param_path)
        # log_file_name = "./log/" + now + "(" + str(args.engine_model) + ").log"
        log_file_name = log_path + now + "(" + str(args.engine_model) + ").log"
        init_log(log_file_name=log_file_name)
        if os.path.exists(args.train_dir):
            pass
        else:
            os.makedirs(args.train_dir)
        run_CGNTag(args=args)

    elif os.path.isdir(param_path):
        list_dir = os.listdir(param_path)
        list_dir.sort(key=lambda x: int(x[33]))
        print(list_dir)
        for file in list_dir:
            one_param_path = os.path.join(param_path, file)
            if os.path.isfile(one_param_path):
                now = datetime.datetime.now().strftime("%Y%m%d%H%M")
                args = init_args(one_param_path)
                log_file_name = log_path + file+"_" + now + "(" + str(args.engine_model) + ").log"
                init_log(log_file_name=log_file_name)
                if os.path.exists(args.train_dir):
                    pass
                else:
                    os.makedirs(args.train_dir)

                run_CGNTag(args=args)