from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
from model import GridCell
import os
from data_io import Data_Generator
from utils import *
#from path_planning import Path_planning, perform_path_planning
from matplotlib import pyplot as plt
from matplotlib import cm
import math
import argparse
from scipy.io import savemat
from mayavi.mlab import *
from mpl_toolkits.mplot3d import axes3d, Axes3D



# tf.set_random_seed(1234)
# np.random.seed(0)

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument('--batch_size', type=int, default=1000000, help='Batch size of training images')  # 1000000
parser.add_argument('--num_epochs', type=int, default=20000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.03, help='Initial learning rate for descriptor')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')

# simulated data parameters
parser.add_argument('--place_size', type=float, default=1.0, help='Size of the square place')
parser.add_argument('--max_vel2', type=float, default=5.77, help='maximum  of velocity in loss2, measure in grids')
parser.add_argument('--min_vel2', type=float, default=1, help='minimum of velocity in loss2, measure in grids')
parser.add_argument('--dtype1', type=int, default=1, help='type of loss1')
parser.add_argument('--shape', type=str, default='square', help='Shape of the area')

# model parameters
parser.add_argument('--sigma', metavar='N', type=float, nargs='+', default=[0.08], help='sd of gaussian kernel')
parser.add_argument('--place_dim', type=int, default=64000, help='Dimensions of place, should be N^3') # 10
parser.add_argument('--num_group', type=int, default=1, help='Number of groups of grid cells') #16
parser.add_argument('--block_size', type=int, default=8, help='Size of each block')
parser.add_argument('--iter', type=int, default=0, help='Number of iter')
parser.add_argument('--lamda', type=float, default=0.1, help='Hyper parameter to balance two loss terms')
parser.add_argument('--GandE', type=float, default=1.0, help='1: Gaussian kernel; 0: Exponential kernel')
parser.add_argument('--lamda2', type=float, default=5000, help='Hyper parameter to balance two loss terms')
parser.add_argument('--lamda3', type=float, default=5.0, help='Hyper parameter to balance two loss terms')
parser.add_argument('--motion_type', type=str, default='continuous', help='True if in testing mode')
parser.add_argument('--num_step', type=int, default=1, help='Number of steps in path integral')
parser.add_argument('--save_memory', type=bool, default=False, help='True if in testing mode')

# parameters for single block tuning
parser.add_argument('--single_block', type=bool, default=True, help='True if in testing mode')
parser.add_argument('--alpha', type=float, default=72.0, help='scale parameter used in single block scenario')

# utils train
parser.add_argument('--output_dir', type=str, default='test', help='The output directory for saving results')
parser.add_argument('--err_dir', type=str, default=None, help='The output directory for saving results')
parser.add_argument('--log_file', type=str, default='con_test.txt', help='The output directory for saving results')
parser.add_argument('--log_step', type=int, default=50, help='Number of mini batches to save output results')

# utils test
parser.add_argument('--mode', type=str, default='0', help='0: training / 1: visualizing')
parser.add_argument('--test_num', type=int, default=5, help='Number of testing steps used in path integral')
parser.add_argument('--project_to_point', type=bool, default=False, help='True if in testing path integral mode')
parser.add_argument('--ckpt', type=str, default='model.ckpt-3299', help='Checkpoint path to load')
parser.add_argument('--num_testing_path_integral', type=int, default=1000, help='Number of testing cases for path integral')
parser.add_argument('--gpu', type=str, default='1', help='Which gpu to use')

FLAGS = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


def train(model, sess, output_dir):
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'model')

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)

    # build model
    model.build_model()
    model.path_integral(FLAGS.test_num)
    # planning_model = Path_planning(model)

    lamda_list = np.linspace(FLAGS.lamda, FLAGS.lamda, FLAGS.num_epochs)
    # initialize training
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=20)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    # make graph immutable
    tf.get_default_graph().finalize()

    # store graph in protobuf
    with open(model_dir + '/graph.proto', 'w') as f:
        f.write(str(tf.get_default_graph().as_graph_def()))

    data_generator = Data_Generator(max=FLAGS.place_size, num_interval=model.num_interval, shape=model.shape)

    # train
    start_time = time.time()
    for epoch in range(FLAGS.num_epochs):
        if epoch < FLAGS.iter:
           lamda_list[epoch] = 0
        place_pair1 = data_generator.generate_3d(FLAGS.batch_size, dtype=FLAGS.dtype1)
        place_seq2 = data_generator.generate_3d(FLAGS.batch_size, max_vel=model.max_vel2, min_vel=model.min_vel2,
                                             num_step=model.num_step, dtype=2, motion_type=model.motion_type)
        alpha = sess.run(model.alpha)
        place_seq3 = []
        if epoch < 0:
            place_seq3 = data_generator.generate_3d(FLAGS.batch_size, max_vel=model.max_vel2, num_step=1, dtype=2)['seq']
            place_seq3 = np.tile(np.expand_dims(place_seq3, axis=1), [1, model.num_group, 1, 1])
        else:
            for block_idx in range(model.num_group):
                # max_vel = 3.0
                max_vel = min(np.sqrt(1.5 / alpha[block_idx]) / model.interval_length, 10)
                place_seqs = data_generator.generate_3d(FLAGS.batch_size, max_vel=max_vel, num_step=1, dtype=2)['seq']
                assert len(place_seqs) == FLAGS.batch_size
                place_seq3.append(place_seqs)
            place_seq3 = np.stack(place_seq3, axis=1)

        loss_avg, loss1_avg, loss2_avg, reg_avg, loss3_avg, loss4_avg = [], [], [], [], [], []


        # update weights
        feed_dict = dict()

        feed_dict.update({model.place_before1: place_pair1['before'],
                     model.place_after1: place_pair1['after'],
                     model.vel1: place_pair1['vel'],
                     model.place_seq2: place_seq2['seq'],
                     model.place_seq3: place_seq3,
                     model.lamda: lamda_list[epoch]})

        feed_dict[model.vel2] = place_seq2['vel'] if model.motion_type == 'continuous' \
            else place_seq2['vel_idx']

        summary, loss, loss1, loss2, reg, loss3, loss4, dp1, dp2 = sess.run([model.summary_op, model.loss, model.loss1,
                                                                 model.loss2, model.reg, model.loss3, model.loss4,
                                                                 model.dp1, model.dp2, model.loss_update,
                                                                 model.apply_grads], feed_dict=feed_dict)[:9]

        # regularize weights
        if epoch > 8000 and not model.single_block:
            sess.run(model.norm_grads)

        loss_avg.append(loss)
        loss1_avg.append(loss1)
        loss2_avg.append(loss2)
        reg_avg.append(reg)
        loss3_avg.append(loss3)
        loss4_avg.append(loss4)

        writer.add_summary(summary, epoch)
        writer.flush()
        if epoch % 10 == 0:
            loss_avg, loss1_avg, loss2_avg, loss3_avg, loss4_avg, reg_avg = np.mean(np.asarray(loss_avg)), \
                                                                            np.mean(np.asarray(loss1_avg)), \
                                                                            np.mean(np.asarray(loss2_avg)), \
                                                                            np.mean(np.asarray(loss3_avg)), \
                                                                            np.mean(np.asarray(loss4_avg)), \
                                                                            np.mean(np.asarray(reg_avg))
            #I2 = sess.run(model.I2)
            end_time = time.time()
            print(alpha)
            print('#{:s} Epoch #{:d}, loss: {:.4f}, loss1: {:.4f}, loss2: {:.4f}, loss3: {:.4f}, reg: {:.4f}, time: {:.2f}s'
                  .format(output_dir, epoch, loss_avg, loss1_avg, loss2_avg, loss3_avg, reg_avg, end_time - start_time))

            start_time = time.time()

        syn_dir = os.path.join(output_dir, 'syn')
        if epoch == 0 or (epoch + 1) % FLAGS.log_step == 0 or epoch == FLAGS.num_epochs - 1:
            saver.save(sess, "%s/%s" % (model_dir, 'model.ckpt'), global_step=epoch)
            if not tf.gfile.Exists(syn_dir):
                tf.gfile.MakeDirs(syn_dir)

            visualize_3D_grid_cell(model, sess, syn_dir, epoch)

def visualize_3D_grid_cell(model, sess, test_dir, epoch=0, slice_to_show=10):
    # only showing one 2D slice of the 3D grid patterns
    weights_A_value = sess.run(model.weights_A)
    if not tf.gfile.Exists(test_dir):
        tf.gfile.MakeDirs(test_dir)
    np.save(os.path.join(test_dir, 'weights.npy'), weights_A_value)

    # print out A
    weights_A_value_transform = weights_A_value.transpose(3, 0, 1, 2)
    # fig_sz = np.ceil(np.sqrt(len(weights_A_value_transform)))

    plt.figure(figsize=(model.block_size, model.num_group))
    for i in range(len(weights_A_value_transform)):
        weight_to_draw = weights_A_value_transform[i]
        plt.subplot(model.num_group, model.block_size, i + 1)

        # showing one slice (2D) of 3D grid patterns
        weight_to_draw_all = weight_to_draw[slice_to_show, :, :]
        draw_heatmap_2D(weight_to_draw_all, vmin=weight_to_draw_all.min(), vmax=weight_to_draw_all.max())

    plt.savefig(os.path.join(test_dir, '3D_patterns_epoch_' + str(epoch) + '.png'))


def main(_):
    model = GridCell(FLAGS)

    with tf.Session() as sess:
        if FLAGS.mode == "1":  # visualize weights
            # load model
            assert FLAGS.ckpt is not None, 'no checkpoint provided.'
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join('output', FLAGS.output_dir, 'model', FLAGS.ckpt))
            print('Loading checkpoint {}.'.format(os.path.join('output', FLAGS.output_dir, 'model', FLAGS.ckpt)))
            test_dir = os.path.join('output', FLAGS.output_dir, 'test')
            #visualize(model, sess, test_dir)
            visualize_3D_grid_cell(model, sess, test_dir)

        elif FLAGS.mode == "0":  # training
            train(model, sess, os.path.join('output', FLAGS.output_dir))
        else:
            raise NotImplementedError


if __name__ == '__main__':
    tf.app.run()
