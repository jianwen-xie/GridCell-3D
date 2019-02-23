from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import scipy.misc
import itertools
from math import pow
import seaborn as sns
import matplotlib.pyplot as plt
from utils import generate_vel_list_3d


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class Data_Generator(object):
    """
    Generate one dimensional simulated data.
    Randomly sample \mu from uniform distribution.
    Velocity is fixed.
    Place vector is generated from a Gaussian distribution.
    """
    def __init__(self, num_interval=1000, min=0, max=1, shape="square", to_use_3D_map=False):
        """
        Sigma is the variance in the Gaussian distribution.
        """        
        self.num_interval = num_interval
        self.min, self.max = min, max
        self.interval_length = (self.max - self.min) / (self.num_interval - 1)
        self.shape = shape

    def generate(self, num_data, max_vel=3, min_vel=0, num_step=1, dtype=2, test=False, visualize=False, motion_type='continuous'):
        
        if dtype == 1:
            place_pair = self.generate_two_dim_multi_type1(num_data)
        elif dtype == 2:
            place_pair = self.generate_two_dim_multi_type2(num_data, max_vel, min_vel, num_step,
                                                           test=test, visualize=visualize, motion_type=motion_type)
        else:
            raise NotImplementedError

        return place_pair    
    
    def generate_3d(self, num_data, max_vel=3, min_vel=0, num_step=1, dtype=2, test=False, visualize=False, motion_type='continuous'):

        if dtype == 1:
            place_pair = self.generate_three_dim_multi_type1(num_data)
        elif dtype == 2:
            place_pair = self.generate_three_dim_multi_type2(num_data, max_vel, min_vel, num_step,
                                                           test=test, visualize=visualize, motion_type=motion_type)
        else:
            raise NotImplementedError

        return place_pair

    def generate_two_dim_multi_type1(self, num_data):
        if self.shape == "square":
            mu_before = np.random.random(size=[num_data, 2]) * (self.num_interval - 1)
            mu_after = np.random.random(size=[num_data, 2]) * (self.num_interval - 1)
        elif self.shape == "circle":
            mu_seq = np.random.random(size=[num_data * 3, 2])
            select_idx = np.where(np.sqrt((mu_seq[:, 0] - 0.5) ** 2 + (mu_seq[:, 1] - 0.5) ** 2) < 0.5)[0]
            mu_seq = mu_seq[select_idx[:num_data * 2]]
            mu_before = mu_seq[:num_data] * (self.num_interval - 1)
            mu_after = mu_seq[num_data:] * (self.num_interval - 1)
        elif self.shape == "triangle":
            mu_seq = np.random.random(size=[int(num_data * 4.2), 2])
            x, y = mu_seq[:, 0], mu_seq[:, 1]
            select_idx = np.where((x + 2 * y > 1) * (-x + 2 * y < 1))[0]
            mu_seq = mu_seq[select_idx[:num_data * 2]]
            mu_before = mu_seq[:num_data] * (self.num_interval - 1)
            mu_after = mu_seq[num_data:] * (self.num_interval - 1)
        else:
            raise NotImplementedError

        vel = np.sqrt(np.sum((mu_after - mu_before) ** 2, axis=1)) * self.interval_length
        place_pair = {'before': mu_before, 'after': mu_after, 'vel': vel}
        assert len(mu_before) == num_data
        return place_pair


    def generate_three_dim_multi_type1(self, num_data):
        if self.shape == "square":
            mu_before = np.random.random(size=[num_data, 3]) * (self.num_interval - 1)
            mu_after = np.random.random(size=[num_data, 3]) * (self.num_interval - 1)

        else:
            raise NotImplementedError

        vel = np.sqrt(np.sum((mu_after - mu_before) ** 2, axis=1)) * self.interval_length
        place_pair = {'before': mu_before, 'after': mu_after, 'vel': vel}
        assert len(mu_before) == num_data
        return place_pair


    def generate_two_dim_multi_type2(self, num_data, max_vel, min_vel, num_step, motion_type, test=False, visualize=False):
        """sample discretized motions and corresponding place pairs"""
        vel_idx = None
        if not test and motion_type == 'discrete':
            velocity = generate_vel_list(max_vel)
            num_vel = len(velocity)
            if pow(num_vel, num_step) < num_data:
                vel_list = np.asarray(list(itertools.product(np.arange(num_vel), repeat=num_step)))
                num_vel_list = len(vel_list)

                div, rem = num_data // num_vel_list, num_data % num_vel_list
                vel_idx = np.vstack((np.tile(vel_list, [div, 1]), vel_list[np.random.choice(num_vel_list, size=rem)]))
                np.random.shuffle(vel_idx)
            else:
                vel_idx = np.random.choice(num_vel, size=[num_data, num_step])

            vel_grid = np.take(velocity, vel_idx, axis=0)
            vel = vel_grid * self.interval_length

            vel_grid_cumsum = np.cumsum(vel_grid, axis=1)
            mu_max = np.fmin(self.num_interval, np.min(self.num_interval - vel_grid_cumsum, axis=1))
            mu_min = np.fmax(0, np.max(-vel_grid_cumsum, axis=1))
            mu_start = np.expand_dims(np.random.random(size=(num_data, 2)) * (mu_max - 1 - mu_min) + mu_min, axis=1)
            # mu_start = np.random.sample(size=[num_data, 2])
            # mu_start = np.expand_dims(np.round(mu_start * (mu_max - mu_min) + mu_min - 0.5), axis=1)
            mu_seq = np.concatenate((mu_start, mu_start + vel_grid_cumsum), axis=1)
        elif not test:
            if self.shape == "square":
                num_data_sample = num_data
            elif self.shape == "circle":
                num_data_sample = int(num_data * 1.5)
            elif self.shape == "triangle":
                num_data_sample = int(num_data * 4)
            else:
                raise NotImplementedError

            theta = np.random.random(size=(num_data_sample, num_step)) * 2 * np.pi - np.pi
            length = np.sqrt(np.random.random(size=(num_data_sample, num_step))) * (max_vel - min_vel) + min_vel
            x = length * np.cos(theta)
            y = length * np.sin(theta)
            vel_seq = np.concatenate((np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)), axis=-1)

            vel_seq_cumsum = np.cumsum(vel_seq, axis=1)
            mu_max = np.fmin(self.num_interval - 1, np.min(self.num_interval - 1 - vel_seq_cumsum, axis=1))
            mu_min = np.fmax(0, np.max(-vel_seq_cumsum, axis=1))
            start = np.random.random(size=(num_data_sample, 2)) * (mu_max - mu_min) + mu_min
            start = np.expand_dims(start, axis=1)

            mu_seq = np.concatenate((start, start + vel_seq), axis=1)
            vel = vel_seq * self.interval_length
            if self.shape == "circle":
                mu_seq_len = mu_seq * self.interval_length
                select_idx = np.sqrt((mu_seq_len[:, :, 0] - 0.5) ** 2 + (mu_seq_len[:, :, 1] - 0.5) ** 2) > 0.5
                select_idx = np.where(np.sum(select_idx, axis=1) == 0)[0]
                mu_seq = mu_seq[select_idx[:num_data]]
                vel = vel[select_idx[:num_data]]
            elif self.shape == "triangle":
                mu_seq_len = mu_seq * self.interval_length
                x, y = mu_seq_len[:, :, 0], mu_seq_len[:, :, 1]
                select_idx = (x + 2 * y > 1) * (-x + 2 * y < 1)
                select_idx = np.where(np.sum(select_idx, axis=1) == num_step + 1)[0]
                mu_seq = mu_seq[select_idx[:num_data]]
                vel = vel[select_idx[:num_data]]
        else:
            velocity = generate_vel_list(max_vel, min_vel)
            num_vel = len(velocity)
            if visualize:
                mu_start = np.reshape([20, 20], newshape=(1, 1, 2))
                vel_pool = np.where((velocity[:, 0] >= -1) & (velocity[:, 1] >= -1))
                vel_idx = np.random.choice(vel_pool[0], size=[num_data * 10, num_step])

                vel_grid_cumsum = np.cumsum(np.take(velocity, vel_idx, axis=0), axis=1)
                mu_seq = np.concatenate((np.tile(mu_start, [num_data * 10, 1, 1]), vel_grid_cumsum + mu_start), axis=1)
                mu_seq_new, vel_idx_new = [], []
                for i in range(len(mu_seq)):
                    mu_seq_sub = mu_seq[i]
                    if len(np.unique(mu_seq_sub, axis=0)) == len(mu_seq_sub):
                        mu_seq_new.append(mu_seq[i])
                        vel_idx_new.append(vel_idx[i])
                mu_seq, vel_idx = np.stack(mu_seq_new, axis=0), np.stack(vel_idx_new, axis=0)
                mu_seq_rs = np.reshape(mu_seq, [-1, (num_step + 1) * 2])
                select_idx = np.where(np.sum(mu_seq_rs >= self.num_interval, axis=1) == 0)[0][:num_data]
                vel_idx = vel_idx[select_idx]
                mu_seq = mu_seq[select_idx]
                vel = np.take(velocity, vel_idx, axis=0) * self.interval_length
            else:
                vel_idx = np.random.choice(num_vel, size=[num_data * 20, num_step])
                vel_grid_cumsum = np.cumsum(np.take(velocity, vel_idx, axis=0), axis=1)
                mu_max = np.fmin(self.num_interval, np.min(self.num_interval - vel_grid_cumsum, axis=1))
                mu_min = np.fmax(0, np.max(-vel_grid_cumsum, axis=1))

                select_idx = np.where(np.sum(mu_max <= mu_min, axis=1) == 0)[0][:num_data]
                vel_idx, vel_grid_cumsum = vel_idx[select_idx], vel_grid_cumsum[select_idx]
                vel_grid = np.take(velocity, vel_idx, axis=0)
                mu_max, mu_min = mu_max[select_idx], mu_min[select_idx]
                mu_start = np.random.sample(size=[num_data, 2])
                mu_start = np.expand_dims(np.round(mu_start * (mu_max - mu_min) + mu_min - 0.5), axis=1)
                mu_seq = np.concatenate((mu_start, mu_start + vel_grid_cumsum), axis=1)
                vel = vel_grid * self.interval_length

        # sns.distplot(vel, rug=True, hist=False)
        # plt.show()
        assert len(mu_seq) == num_data
        place_seq = {'seq': mu_seq, 'vel': vel, 'vel_idx': vel_idx}
        return place_seq


    def generate_three_dim_multi_type2(self, num_data, max_vel, min_vel, num_step, motion_type, test=False, visualize=False):
        """sample discretized motions and corresponding place pairs"""
        vel_idx = None
        if not test and motion_type == 'discrete':
            velocity = generate_vel_list_3d(max_vel)
            num_vel = len(velocity)
            if pow(num_vel, num_step) < num_data:
                vel_list = np.asarray(list(itertools.product(np.arange(num_vel), repeat=num_step)))
                num_vel_list = len(vel_list)

                div, rem = num_data // num_vel_list, num_data % num_vel_list
                vel_idx = np.vstack((np.tile(vel_list, [div, 1]), vel_list[np.random.choice(num_vel_list, size=rem)]))
                np.random.shuffle(vel_idx)
            else:
                vel_idx = np.random.choice(num_vel, size=[num_data, num_step])

            vel_grid = np.take(velocity, vel_idx, axis=0)
            vel = vel_grid * self.interval_length

            vel_grid_cumsum = np.cumsum(vel_grid, axis=1)
            mu_max = np.fmin(self.num_interval, np.min(self.num_interval - vel_grid_cumsum, axis=1))
            mu_min = np.fmax(0, np.max(-vel_grid_cumsum, axis=1))
            mu_start = np.expand_dims(np.random.random(size=(num_data, 2)) * (mu_max - 1 - mu_min) + mu_min, axis=1)
            # mu_start = np.random.sample(size=[num_data, 2])
            # mu_start = np.expand_dims(np.round(mu_start * (mu_max - mu_min) + mu_min - 0.5), axis=1)
            mu_seq = np.concatenate((mu_start, mu_start + vel_grid_cumsum), axis=1)
        elif not test:
            if self.shape == "square":
                num_data_sample = num_data
            else:
                raise NotImplementedError

            #theta = np.random.random(size=(num_data_sample, num_step)) * 2 * np.pi - np.pi
            #length = np.sqrt(np.random.random(size=(num_data_sample, num_step))) * (max_vel - min_vel) + min_vel
            #x = length * np.cos(theta)
            #y = length * np.sin(theta)
            #vel_seq = np.concatenate((np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)), axis=-1)

            x1 = np.random.standard_normal(size=(num_data_sample, num_step))
            y1 = np.random.standard_normal(size=(num_data_sample, num_step))
            z1 = np.random.standard_normal(size=(num_data_sample, num_step))
            v = np.sqrt(x1**2 + y1 ** 2 + z1 ** 2)           
            length = np.cbrt(np.random.random(size=(num_data_sample, num_step))) * (max_vel - min_vel) + min_vel
            
            x = length * x1 / v
            y = length * y1 / v
            z = length * z1 / v
            vel_seq = np.concatenate((np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(z, axis=-1)), axis=-1)

            # from matplotlib import pyplot
            # from mpl_toolkits.mplot3d import Axes3D
            # fig = pyplot.figure()
            # ax = Axes3D(fig)
            # ax.scatter(x[:30000], y[0:30000], z[0:30000])
            # pyplot.show()


            vel_seq_cumsum = np.cumsum(vel_seq, axis=1)
            mu_max = np.fmin(self.num_interval - 1, np.min(self.num_interval - 1 - vel_seq_cumsum, axis=1))
            mu_min = np.fmax(0, np.max(-vel_seq_cumsum, axis=1))
            start = np.random.random(size=(num_data_sample, 3)) * (mu_max - mu_min) + mu_min
            start = np.expand_dims(start, axis=1)

            mu_seq = np.concatenate((start, start + vel_seq), axis=1)
            vel = vel_seq * self.interval_length
            if self.shape == "circle":
                mu_seq_len = mu_seq * self.interval_length
                select_idx = np.sqrt((mu_seq_len[:, :, 0] - 0.5) ** 2 + (mu_seq_len[:, :, 1] - 0.5) ** 2) > 0.5
                select_idx = np.where(np.sum(select_idx, axis=1) == 0)[0]
                mu_seq = mu_seq[select_idx[:num_data]]
                vel = vel[select_idx[:num_data]]
            elif self.shape == "triangle":
                mu_seq_len = mu_seq * self.interval_length
                x, y = mu_seq_len[:, :, 0], mu_seq_len[:, :, 1]
                select_idx = (x + 2 * y > 1) * (-x + 2 * y < 1)
                select_idx = np.where(np.sum(select_idx, axis=1) == num_step + 1)[0]
                mu_seq = mu_seq[select_idx[:num_data]]
                vel = vel[select_idx[:num_data]]
        else:
            velocity = generate_vel_list_3d(max_vel, min_vel)
            num_vel = len(velocity)
            if visualize:
                mu_start = np.reshape([20, 20, 20], newshape=(1, 1, 1, 3))
                vel_pool = np.where((velocity[:, 0] >= -1) & (velocity[:, 1] >= -1) & (velocity[:, 2] >= -1))
                vel_idx = np.random.choice(vel_pool[0], size=[num_data * 10, num_step])

                vel_grid_cumsum = np.cumsum(np.take(velocity, vel_idx, axis=0), axis=1)
                mu_seq = np.concatenate((np.tile(mu_start, [num_data * 10, 1, 1, 1]), vel_grid_cumsum + mu_start), axis=1)
                mu_seq_new, vel_idx_new = [], []
                for i in range(len(mu_seq)):
                    mu_seq_sub = mu_seq[i]
                    if len(np.unique(mu_seq_sub, axis=0)) == len(mu_seq_sub):
                        mu_seq_new.append(mu_seq[i])
                        vel_idx_new.append(vel_idx[i])
                mu_seq, vel_idx = np.stack(mu_seq_new, axis=0), np.stack(vel_idx_new, axis=0)
                mu_seq_rs = np.reshape(mu_seq, [-1, (num_step + 1) * 2])
                select_idx = np.where(np.sum(mu_seq_rs >= self.num_interval, axis=1) == 0)[0][:num_data]
                vel_idx = vel_idx[select_idx]
                mu_seq = mu_seq[select_idx]
                vel = np.take(velocity, vel_idx, axis=0) * self.interval_length
            else:
                vel_idx = np.random.choice(num_vel, size=[num_data * 20, num_step])
                vel_grid_cumsum = np.cumsum(np.take(velocity, vel_idx, axis=0), axis=1)
                mu_max = np.fmin(self.num_interval, np.min(self.num_interval - vel_grid_cumsum, axis=1))
                mu_min = np.fmax(0, np.max(-vel_grid_cumsum, axis=1))

                select_idx = np.where(np.sum(mu_max <= mu_min, axis=1) == 0)[0][:num_data]
                vel_idx, vel_grid_cumsum = vel_idx[select_idx], vel_grid_cumsum[select_idx]
                vel_grid = np.take(velocity, vel_idx, axis=0)
                mu_max, mu_min = mu_max[select_idx], mu_min[select_idx]
                mu_start = np.random.sample(size=[num_data, 3])
                mu_start = np.expand_dims(np.round(mu_start * (mu_max - mu_min) + mu_min - 0.5), axis=1)
                mu_seq = np.concatenate((mu_start, mu_start + vel_grid_cumsum), axis=1)
                vel = vel_grid * self.interval_length

        # sns.distplot(vel, rug=True, hist=False)
        # plt.show()
        assert len(mu_seq) == num_data
        place_seq = {'seq': mu_seq, 'vel': vel, 'vel_idx': vel_idx}
        return place_seq


# class Data_Generator(object):
#     """
#     Generate one dimensional simulated data.
#     Randomly sample \mu from uniform distribution.
#     Velocity is fixed.
#     Place vector is generated from a Gaussian distribution.
#     """
#     def __init__(self, num_interval=1000, min=0, max=1):
#         """
#         Sigma is the variance in the Gaussian distribution.
#         """
#         self.num_interval = num_interval
#         self.min, self.max = min, max
#         self.interval_length = (self.max - self.min) / (self.num_interval - 1)
#
#     def generate(self, num_data, velocity=None, num_step=1, dtype=2, test=False, visualize=False):
#         if dtype == 1:
#             place_pair = self.generate_two_dim_multi_type1(num_data)
#         elif dtype == 2:
#             place_pair = self.generate_two_dim_multi_type2(num_data, velocity, num_step, test=test, visualize=visualize)
#         elif dtype == 4:
#             place_pair = self.generate_two_dim_multi_type4(num_data)
#         else:
#             raise NotImplementedError
#
#         return place_pair
#
#     def generate_two_dim_multi_type1(self, num_data):
#         mu_before = np.random.choice(self.num_interval, size=[num_data, 2])
#         mu_after = np.random.choice(self.num_interval, size=[num_data, 2])
#
#         vel = np.sqrt(np.sum((mu_after - mu_before) ** 2, axis=1)) * self.interval_length
#
#         place_pair = {'before': mu_before, 'after': mu_after, 'vel': vel}
#
#         return place_pair
#
#     def generate_two_dim_multi_type2(self, num_data, velocity, num_step, test=False, visualize=False):
#         """sample discretized motions and corresponding place pairs"""
#         num_vel = len(velocity)
#         if not test:
#             if pow(num_vel, num_step) < num_data:
#                 vel_list = np.asarray(list(itertools.product(np.arange(num_vel), repeat=num_step)))
#                 num_vel_list = len(vel_list)
#
#                 div, rem = num_data // num_vel_list, num_data % num_vel_list
#                 vel_idx = np.vstack((np.tile(vel_list, [div, 1]), vel_list[np.random.choice(num_vel_list, size=rem)]))
#                 np.random.shuffle(vel_idx)
#             else:
#                 vel_idx = np.random.choice(num_vel, size=[num_data, num_step])
#
#             vel_grid = np.take(velocity, vel_idx, axis=0)
#             vel = vel_grid * self.interval_length
#
#             vel_grid_cumsum = np.cumsum(vel_grid, axis=1)
#             mu_max = np.fmin(self.num_interval, np.min(self.num_interval - vel_grid_cumsum, axis=1))
#             mu_min = np.fmax(0, np.max(-vel_grid_cumsum, axis=1))
#             mu_start = np.random.sample(size=[num_data, 2])
#             mu_start = np.expand_dims(np.round(mu_start * (mu_max - mu_min) + mu_min - 0.5), axis=1)
#             mu_seq = np.concatenate((mu_start, mu_start + vel_grid_cumsum), axis=1)
#         else:
#             if visualize:
#                 mu_start = np.reshape([4, 4], newshape=(1, 1, 2))
#                 vel_pool = np.where((velocity[:, 0] >= -1) & (velocity[:, 1] >= -1))
#                 vel_idx = np.random.choice(vel_pool[0], size=[num_data * 10, num_step])
#
#                 vel_grid_cumsum = np.cumsum(np.take(velocity, vel_idx, axis=0), axis=1)
#                 mu_seq = np.concatenate((np.tile(mu_start, [num_data * 10, 1, 1]), vel_grid_cumsum + mu_start), axis=1)
#                 mu_seq_new, vel_idx_new = [], []
#                 for i in range(len(mu_seq)):
#                     mu_seq_sub = mu_seq[i]
#                     if len(np.unique(mu_seq_sub, axis=0)) == len(mu_seq_sub):
#                         mu_seq_new.append(mu_seq[i])
#                         vel_idx_new.append(vel_idx[i])
#                 mu_seq, vel_idx = np.stack(mu_seq_new, axis=0), np.stack(vel_idx_new, axis=0)
#                 mu_seq_rs = np.reshape(mu_seq, [-1, (num_step + 1) * 2])
#                 select_idx = np.where(np.sum(mu_seq_rs >= self.num_interval, axis=1) == 0)[0][:num_data]
#                 vel_idx = vel_idx[select_idx]
#                 mu_seq = mu_seq[select_idx]
#                 vel = np.take(velocity, vel_idx, axis=0) * self.interval_length
#             else:
#                 vel_idx = np.random.choice(num_vel, size=[num_data * 20, num_step])
#                 vel_grid_cumsum = np.cumsum(np.take(velocity, vel_idx, axis=0), axis=1)
#                 mu_max = np.fmin(self.num_interval, np.min(self.num_interval - vel_grid_cumsum, axis=1))
#                 mu_min = np.fmax(0, np.max(-vel_grid_cumsum, axis=1))
#
#                 select_idx = np.where(np.sum(mu_max <= mu_min, axis=1) == 0)[0][:num_data]
#                 vel_idx, vel_grid_cumsum = vel_idx[select_idx], vel_grid_cumsum[select_idx]
#                 vel_grid = np.take(velocity, vel_idx, axis=0)
#                 mu_max, mu_min = mu_max[select_idx], mu_min[select_idx]
#                 mu_start = np.random.sample(size=[num_data, 2])
#                 mu_start = np.expand_dims(np.round(mu_start * (mu_max - mu_min) + mu_min - 0.5), axis=1)
#                 mu_seq = np.concatenate((mu_start, mu_start + vel_grid_cumsum), axis=1)
#                 vel = vel_grid * self.interval_length
#
#         # sns.distplot(vel, rug=True, hist=False)
#         # plt.show()
#
#         place_seq = {'seq': mu_seq, 'vel': vel, 'vel_idx': vel_idx}
#         return place_seq
#
#     def generate_two_dim_multi_type3(self, num_data, max_vel, num_step, test=False):
#         """sample discretized motions and corresponding place pairs"""
#         max_vel = max_vel * self.interval_length
#         if not test:
#             r = np.sqrt(np.random.random(size=[num_data, num_step])) * max_vel
#             theta = np.random.uniform(low=-np.pi, high=np.pi, size=[num_data, num_step])
#
#             vel = np.zeros(shape=(num_data, num_step, 2), dtype=float)
#             vel[:, :, 0] = r * np.cos(theta)
#             vel[:, :, 1] = r * np.sin(theta)
#             vel_cumsum = np.cumsum(vel, axis=1)
#
#             mu_max = np.fmin(1, np.min(1 - vel_cumsum, axis=1))
#             mu_min = np.fmax(0, np.max(- vel_cumsum, axis=1))
#             mu_start = np.random.random(size=(num_data, 2)) * (mu_max - mu_min) + mu_min
#             mu_start = np.expand_dims(mu_start, axis=1)
#
#             mu_seq = np.concatenate((mu_start, mu_start + vel_cumsum), axis=1) / self.interval_length
#         else:
#             if num_data == 1:
#
#                 mu_start = np.reshape([6, 6], newshape=(1, 1, 2)) * self.interval_length
#                 r = np.sqrt(np.random.random(size=[num_data * 10, num_step])) * max_vel
#                 theta = np.random.uniform(low=-np.pi, high=np.pi, size=[num_data * 10, num_step])
#
#                 vel = np.zeros(shape=(num_data * 10, num_step, 2), dtype=float)
#                 vel[:, :, 0] = r * np.cos(theta)
#                 vel[:, :, 1] = r * np.sin(theta)
#                 vel[np.where(vel <= 0)[0]] = vel[np.where(vel <= 0)[0]] * 0.3
#                 vel_cumsum = np.cumsum(vel, axis=1)
#
#                 mu_seq = np.concatenate((mu_start, mu_start + vel_cumsum), axis=1) / self.interval_length
#                 select_idx = np.where(np.sum(mu_seq > self.num_interval - 1, axis=1) == 0)[0][0]
#                 mu_seq = np.expand_dims(mu_seq[select_idx], axis=0)
#                 vel = np.expand_dims(vel[select_idx], axis=0)
#             else:
#                 r = np.sqrt(np.random.random(size=[num_data * 10, num_step])) * max_vel
#                 theta = np.random.uniform(low=-np.pi, high=np.pi, size=[num_data * 10, num_step])
#
#                 vel = np.zeros(shape=(num_data * 10, num_step, 2), dtype=float)
#                 vel[:, :, 0] = r * np.cos(theta)
#                 vel[:, :, 1] = r * np.sin(theta)
#                 vel_cumsum = np.cumsum(vel, axis=1)
#                 mu_max = np.fmin(1, np.min(1 - vel_cumsum, axis=1))
#                 mu_min = np.fmax(0, np.max(- vel_cumsum, axis=1))
#
#                 select_idx = np.where(mu_max > mu_min)[0][:num_data]
#                 vel, vel_cumsum, mu_min, mu_max = vel[select_idx], vel_cumsum[select_idx], mu_min[select_idx], mu_max[select_idx]
#                 mu_start = np.expand_dims(np.random.random(size=(num_data, 2)) * (mu_max - mu_min) + mu_min, axis=1)
#
#                 mu_seq = np.concatenate((mu_start, mu_start + vel_cumsum), axis=1) / self.interval_length
#
#         place_seq = {'seq': mu_seq, 'vel': vel}
#         return place_seq
#
#     def generate_two_dim_multi_type4(self, num_data):
#         """sample distance by exp distribution"""
#         mu_before = np.random.choice(self.num_interval, size=[num_data * 3, 2])
#         r = np.random.exponential(scale=10, size=[num_data * 3])
#         theta = np.random.uniform(low=0, high=2 * math.pi, size=[num_data * 3])
#
#         mu_after = np.zeros(shape=mu_before.shape, dtype=np.float32)
#         mu_after[:, 0] = mu_before[:, 0] + r * np.cos(theta)
#         mu_after[:, 1] = mu_before[:, 1] + r * np.sin(theta)
#         mu_after = np.around(mu_after).astype(np.int)
#
#         select_id = np.where((mu_after[:, 0] >= 0) & (mu_after[:, 0] <= self.num_interval) &
#                              (mu_after[:, 1] >= 0) & (mu_after[:, 1] <= self.num_interval))[0][:num_data]
#
#         mu_before, mu_after = mu_before[select_id], mu_after[select_id]
#         vel = np.sqrt(np.sum((mu_after - mu_before) ** 2, axis=1)) * self.interval_length
#
#         place_pair = {'before': mu_before, 'after': mu_after, 'vel': vel}
#         return place_pair


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


def cell2img(cell_image, image_size=100, margin_syn=2):
    num_cols = cell_image.shape[1] // image_size
    num_rows = cell_image.shape[0] // image_size
    images = np.zeros((num_cols * num_rows, image_size, image_size, 3))
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = cell_image[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),
                   ic*(image_size+margin_syn):image_size + ic*(image_size+margin_syn),:]
            images[ir*num_cols+ic] = temp
    return images


def clip_by_value(input_, min=0, max=1):
    return np.minimum(max, np.maximum(min, input_))


def img2cell(images, row_num=10, col_num=10, margin_syn=2):
    [num_images, image_size] = images.shape[0:2]
    num_cells = int(math.ceil(num_images / (col_num * row_num)))
    cell_image = np.zeros((num_cells, row_num * image_size + (row_num-1)*margin_syn,
                           col_num * image_size + (col_num-1)*margin_syn, images.shape[-1]))
    for i in range(num_images):
        cell_id = int(math.floor(i / (col_num * row_num)))
        idx = i % (col_num * row_num)
        ir = int(math.floor(idx / col_num))
        ic = idx % col_num
        temp = clip_by_value(np.squeeze(images[i]), -1, 1)
        temp = (temp + 1) / 2 * 255
        temp = clip_by_value(np.round(temp), min=0, max=255)
        if len(temp.shape) == 3:
            gLow = np.min(temp, axis=(0, 1, 2))
            gHigh = np.max(temp, axis=(0, 1, 2))
        elif len(temp.shape) == 2:
            gLow = np.min(temp, axis=(0, 1))
            gHigh = np.max(temp, axis=(0, 1))
        temp = (temp - gLow) / (gHigh - gLow)
        if len(temp.shape) == 2:
            temp = np.expand_dims(temp, axis=2)
        cell_image[cell_id, (image_size+margin_syn)*ir:image_size + (image_size+margin_syn)*ir,
        (image_size+margin_syn)*ic:image_size + (image_size+margin_syn)*ic, :] = temp
    return cell_image


def saveSampleResults(sample_results, filename, col_num=10, margin_syn=2):
    cell_image = img2cell(sample_results, col_num, col_num, margin_syn)
    scipy.misc.imsave(filename, np.squeeze(cell_image))