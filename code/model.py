########################################################################################################################
####                           This file holds all tools and model structure                                        ####
########################################################################################################################

import tensorflow as tf
from tensorflow.contrib import slim
from parameters import GROWTH_RATE, KERNEL_SIZE_DENSE, BOTTLE_OUTPUT, LAYER_PER_BLOCK_DENSE, KERNEL_SIZE_NORMAL, \
    KERNEL_SIZE_POOLING
from tensorflow.python.ops import init_ops


class model_tools:
    @staticmethod
    def lrelu(x, coef=0.2):
        return tf.maximum(x, coef * x)

    @staticmethod
    def l1_loss(x, gt):
        return tf.reduce_mean(tf.abs(x - gt))

    @staticmethod
    def l2_loss(x, gt):
        return tf.reduce_mean(tf.square(x - gt))

    @staticmethod
    def pack_raw(x):
        '''
        input is a raw image or rgb image, and 
        :param x: input image, raw or rgb
        :return: a four channel image with r, g, g, b
        '''
        img_shape = x.shape
        H = img_shape[1]
        W = img_shape[2]
        output = []

        for i in range(x.get_shape().as_list()[0]):
            im = x[i, :, :, :]
            out = tf.concat((im[0:H:2, 0:W:2, :], 
                             im[0:H:2, 1:W:2, :],
                             im[1:H:2, 0:W:2, :],
                             im[1:H:2, 1:W:2, :]
                             ), axis=2)

            output.append(out)
        return tf.stack(output)

    @staticmethod
    def block(x, growth_rate, layers_per_block, kernel_size, reuse, scope_name):
        with tf.variable_scope(scope_name, reuse=reuse):
            conv_in = []
            next_layer = x
            for i in range(1, layers_per_block + 1):
                next_layer = tf.concat(conv_in, axis=3) if i != 1 else next_layer
                conv = slim.conv2d(inputs=next_layer, num_outputs=growth_rate, kernel_size=kernel_size,
                                   rate=1, scope='conv_%d' % i, reuse=reuse, activation_fn=model_tools.lrelu)
                conv_in.append(conv)
            dense = tf.concat(conv_in, axis=3)
            return dense, dense 



class model:
    def __init__(self, input_img, input_isp, gt_img, reuse):
        self.input_img = input_img
        self.input_isp = input_isp
        self.gt_img = gt_img
        self.reuse = reuse
        self.growth_rate = GROWTH_RATE
        self.kernel_size_dense = KERNEL_SIZE_DENSE
        self.kernel_size_pool = KERNEL_SIZE_POOLING
        self.kernel_size = KERNEL_SIZE_NORMAL
        self.layers_per_block = LAYER_PER_BLOCK_DENSE
        self.bottle_output = BOTTLE_OUTPUT

    def branch_1_0(self):
        ################################################################################################################
        ####    Branch_1_0: Input: RawImage Output: fc1: bottle layer output just before  deconv                    ####
        ####                                        helper1: concat of feature map with size H*W for final deconv   ####
        ################################################################################################################
        with tf.variable_scope('branch_1_0', reuse=self.reuse):
            pack_img = model_tools.pack_raw(self.input_img)

            conv_1_0_low = slim.conv2d(inputs=pack_img, num_outputs=128, kernel_size=self.kernel_size,
                                       scope='conv_1_0_low', reuse=self.reuse, activation_fn=model_tools.lrelu)

            dense_1_0, next_in_0 = model_tools.block(conv_1_0_low, self.growth_rate, self.layers_per_block,
                                                     self.kernel_size_dense, self.reuse, 'dense_1_0')
            pool_in_0 = slim.conv2d(inputs=next_in_0, num_outputs=128, kernel_size=self.kernel_size,
                                    stride=2, scope='pool_in_0', reuse=self.reuse, activation_fn=model_tools.lrelu)
            dense_1_1, next_in_1 = model_tools.block(pool_in_0, self.growth_rate, self.layers_per_block,
                                                     self.kernel_size_dense, self.reuse, 'dense_1_1')
            dense_1_2, next_in_2 = model_tools.block(next_in_1, self.growth_rate, self.layers_per_block,
                                                     self.kernel_size_dense, self.reuse, 'dense_1_2')
            bottle_1_0 = slim.conv2d(inputs=tf.concat([dense_1_2, dense_1_1, pool_in_0], axis=3),
                                     num_outputs=128, kernel_size=1, scope='bottle_1_0', reuse=self.reuse,
                                     activation_fn=model_tools.lrelu)
            fc1 = bottle_1_0
            helper1 = tf.concat([conv_1_0_low, dense_1_0], axis=3)
            return fc1, helper1

    def branch_1_1(self, fcin1, helper1):
        ################################################################################################################
        ####    Branch_1_0: Input: RawImage Output: fc1: bottle layer output just before  deconv                    ####
        ####                                        helper1: concat of feature map with size H*W for final deconv   ####
        ################################################################################################################
        with tf.variable_scope('branch_1_1', reuse=self.reuse):
            deconv_1_1_0 = slim.conv2d_transpose(inputs=fcin1, num_outputs=128, kernel_size=[4, 4], stride=2,
                                                 reuse=self.reuse, scope='deconv_1_1_0',
                                                 activation_fn=model_tools.lrelu)
            conv_1_1_0 = slim.conv2d(inputs=deconv_1_1_0, num_outputs=128, kernel_size=self.kernel_size,
                                     scope='conv_1_1_0', reuse=self.reuse, activation_fn=model_tools.lrelu)
            dense_1_0, next_in_0 = model_tools.block(conv_1_1_0, self.growth_rate, self.layers_per_block,
                                                     self.kernel_size_dense, self.reuse, 'dense_1_0')
            bottle_1_1 = slim.conv2d(inputs=tf.concat([helper1, conv_1_1_0, dense_1_0], axis=3),
                                     num_outputs=self.bottle_output, kernel_size=1, scope='bottle_1_1',
                                     reuse=self.reuse,
                                     activation_fn=model_tools.lrelu)
            #conv_1_1_1 = bottle_1_1
            conv_1_1_1 = slim.conv2d(inputs=bottle_1_1, num_outputs=48, kernel_size=self.kernel_size,
                                     scope='conv_1_1_1',
                                     reuse=self.reuse, activation_fn=model_tools.lrelu)
            conv_r = tf.depth_to_space(tf.depth_to_space(conv_1_1_1[:, :, :, :16], 2), 2)
            conv_g = tf.depth_to_space(tf.depth_to_space(conv_1_1_1[:, :, :, 16:32], 2), 2)
            conv_b = tf.depth_to_space(tf.depth_to_space(conv_1_1_1[:, :, :, 32:], 2), 2)
            rgb = tf.concat([conv_r, conv_g, conv_b], axis=3)
            return rgb

    def branch_2_0(self):
        with tf.variable_scope('branch_2_0', reuse=self.reuse):
            conv_2_0_0 = slim.conv2d(inputs=self.input_isp, num_outputs=128, kernel_size=self.kernel_size,
                                     reuse=self.reuse, scope='conv_2_0_0', activation_fn=model_tools.lrelu)
            pool_2_0_0 = slim.avg_pool2d(inputs=conv_2_0_0, kernel_size=2, scope='pool_2_0_0')

            conv_2_0_1 = slim.conv2d(inputs=pool_2_0_0, num_outputs=128, kernel_size=self.kernel_size,
                                     reuse=self.reuse, scope='conv_2_0_1', activation_fn=model_tools.lrelu)
            pool_2_0_1 = slim.avg_pool2d(inputs=conv_2_0_1, kernel_size=2, scope='pool_2_0_1')

            conv_2_0_2 = slim.conv2d(inputs=pool_2_0_1, num_outputs=128, kernel_size=self.kernel_size,
                                     reuse=self.reuse, scope='conv_2_0_2', activation_fn=model_tools.lrelu)
            fc2 = conv_2_0_2
            return fc2, pool_2_0_1, pool_2_0_0

    def branch_2_1(self, fcin2, pool_2_0_1, pool_2_0_0):
        with tf.variable_scope('branch_2_1', reuse=self.reuse):
            deconv_2_1_0 = slim.conv2d_transpose(inputs=tf.concat([fcin2, pool_2_0_1], axis=3), num_outputs=128,
                                                 kernel_size=[4, 4], reuse=self.reuse, stride=2, scope='deconv_2_0_0',
                                                 activation_fn=model_tools.lrelu)
            conv_2_1_3 = slim.conv2d(inputs=deconv_2_1_0, num_outputs=128, kernel_size=self.kernel_size,
                                     reuse=self.reuse, scope='conv_2_0_3', activation_fn=model_tools.lrelu)

            deconv_2_1_1 = slim.conv2d_transpose(inputs=tf.concat([conv_2_1_3, pool_2_0_0], axis=3), num_outputs=128,
                                                 kernel_size=[4, 4], reuse=self.reuse, stride=2, scope='deconv_2_0_1',
                                                 activation_fn=model_tools.lrelu)
            conv_2_1_4 = slim.conv2d(inputs=deconv_2_1_1, num_outputs=128, kernel_size=self.kernel_size,
                                     reuse=self.reuse, scope='deconv_2_0_4', activation_fn=model_tools.lrelu)

            deconv_2_1_2 = slim.conv2d_transpose(inputs=conv_2_1_4, num_outputs=128, kernel_size=[4, 4],
                                                 reuse=self.reuse,
                                                 stride=2, scope='deconv_2_0_2', activation_fn=model_tools.lrelu)
            conv_2_1_5 = slim.conv2d(inputs=deconv_2_1_2, num_outputs=12, kernel_size=self.kernel_size,
                                     reuse=self.reuse,
                                     scope='conv_2_0_5', activation_fn=None)

            color_matrix_per_pixel = conv_2_1_5
            return color_matrix_per_pixel

    def branch_3(self, rgb, color_matrix_per_pixel):
        with tf.variable_scope('branch_3', reuse=self.reuse):

            ############################################################################################################
            ####                                Use perpixel color matrix to rgb                                    ####
            ############################################################################################################
            r_matrix = color_matrix_per_pixel[:, :, :, :3]
            g_matrix = color_matrix_per_pixel[:, :, :, 3:6]
            b_matrix = color_matrix_per_pixel[:, :, :, 6:9]
            r_m = tf.reduce_sum(r_matrix*rgb, axis=3) + color_matrix_per_pixel[:, :, :, 9]
            g_m = tf.reduce_sum(g_matrix * rgb, axis=3) + color_matrix_per_pixel[:, :, :, 10]
            b_m = tf.reduce_sum(b_matrix * rgb, axis=3) + color_matrix_per_pixel[:, :, :, 11]

            r = tf.expand_dims(r_m, axis=3)
            g = tf.expand_dims(g_m, axis=3)
            b = tf.expand_dims(b_m, axis=3)
            srgb = tf.concat([r, g, b], axis=3)
            return srgb

    def build_model(self):
        with tf.variable_scope('main_model', reuse=self.reuse):
            fc1, helper1 = model.branch_1_0(self)
            fc2, pool_2_0_1, pool_2_0_0 = model.branch_2_0(self)
            fin1 = fc1
            fin2 = slim.conv2d(inputs=fc1, num_outputs=128, kernel_size=1, activation_fn=model_tools.lrelu,
                               reuse=self.reuse, scope='w_fin2',
                               weights_initializer=init_ops.zeros_initializer(dtype=tf.float32),
                               biases_initializer=init_ops.zeros_initializer(dtype=tf.float32)) + fc2
            rgb = model.branch_1_1(self, fcin1=fin1, helper1=helper1)
            color_matrix_per_pixel = model.branch_2_1(self, fcin2=fin2, pool_2_0_0=pool_2_0_0,
                                                                pool_2_0_1=pool_2_0_1)
            srgb = model.branch_3(self, rgb=rgb, color_matrix_per_pixel=color_matrix_per_pixel)
            if self.gt_img is not None:
                loss = model_tools.l1_loss(srgb, self.gt_img)
                psnr = tf.reduce_mean(tf.image.psnr(srgb, self.gt_img, max_val=1.0), axis=0)
                ssim = tf.reduce_mean(tf.image.ssim(srgb, self.gt_img, max_val=1.0), axis=0)
            else:
                loss = None
                psnr = None
                ssim = None
            res = srgb

            return loss, psnr, ssim, res




