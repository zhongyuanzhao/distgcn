import os
import numpy as np

def create_result_folder(FLAGS, postfix):
    # test data path
    data_path = FLAGS.datapath

    if FLAGS.greedy == 1:
        greedy_string = "_greedy"
    elif FLAGS.greedy == 2:
        greedy_string = "_greedy_snr{}".format(FLAGS.snr_db)
    else:
        greedy_string = "_" + FLAGS.predict

    if FLAGS.wts_init == 'zeros':
        initstr = "zeros"
    else:
        initstr = ""

    if FLAGS.skip:
        skipstr = "_skip"
    else:
        skipstr = "_no_skip"

    # outputfolder = "./res_{:04d}_{}_{}_{}_wts_prob".format(time_limit, FLAGS.diver_num, FLAGS.diver_out, FLAGS.backoff_prob)
    outputfolder = "./res_{:04d}_{}_{}_{}_{}_{}{}{}_{}".format(FLAGS.timeout, FLAGS.training_set+initstr, FLAGS.diver_num, FLAGS.diver_out, FLAGS.backoff_prob, data_path.split("/")[-1], greedy_string, skipstr, postfix)
    if not os.path.isdir(outputfolder):
        os.makedirs(outputfolder)
    return outputfolder

def find_model_folder(FLAGS, postfix):
    # Copy trained model to /model
    model_origin = "result_{}_deep_ld{}_c{}_l{}_cheb{}_diver{}_{}_{}".format(FLAGS.training_set, FLAGS.feature_size, FLAGS.hidden1, FLAGS.num_layer, FLAGS.max_degree, FLAGS.diver_num, FLAGS.predict, postfix)
    model_origin = os.path.join('./model', model_origin)
    if not FLAGS.snapshot == "":
        model_origin = os.path.join(model_origin, FLAGS.snapshot)
    return model_origin
