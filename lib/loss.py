import numpy as np
import tensorflow as tf
import lib.vgg as vgg
from tensorflow.contrib import slim


def filter_endpoints(end_points):
    """
    Keep only conv layers and reshape into
    [Nl x Ml] matrixes
    """

    d = {}

    for k, v in end_points.items():
        if 'conv' in k:
            el = tf.squeeze(v)
            n_f = el.shape[-1]
            el = tf.reshape(el, (-1, n_f))
            el = tf.transpose(el)
            el = tf.nn.relu(el)
            d[k.split('/')[-1]] = el

    return d


def loss_style(f_im, f_im_target, w):
    """
    Compute style loss

    :f_im: features from the generated image
    :f_im_target: features from the target image
    :w: weighting factors
    """

    loss, weights = [], []
    for key in f_im.keys():
        n, m = f_im[key].get_shape().as_list()
        g_im = tf.matmul(f_im[key], tf.transpose(f_im[key]))
        g_im_target = tf.matmul(f_im_target[key], tf.transpose(f_im_target[key]))
        tmp_loss = tf.nn.l2_loss(g_im_target-g_im)/n/n/m/m/2.
        loss.append(tmp_loss)
        weights.append(w[key])

    weights = np.array(weights)
    # Normalize weighting factors
    weights /= np.sum(weights)
    weights = weights.astype(np.float32)

    return tf.reduce_sum(tf.multiply(weights, loss))


def loss_content(f_im, f_im_target, w):
    """
    Compute content loss

    :f_im: features from the generated image
    :f_im_target: features from the target image
    :w: weighting factors
    """

    loss, weights = [], []
    for key in f_im.keys():
        tmp_loss = tf.nn.l2_loss(f_im_target[key]-f_im[key])
        loss.append(tmp_loss)
        weights.append(w[key])

    weights = np.array(weights)
    # Normalize weighting factors
    weights /= np.sum(weights)
    weights = weights.astype(np.float32)

    return tf.reduce_sum(tf.multiply(weights, loss))


def total_variation_loss(image):
    """
    Return total variation loss

    :image: generated image
    """

    tv_loss = tf.nn.l2_loss(image[:, 1:, :, :]-image[:, :-1, :, :])
    tv_loss += tf.nn.l2_loss(image[:, :, 1:, :]-image[:, :, :-1, :])
    return 2*tv_loss


def get_loss(image, target_content, target_style,
             w_content, w_style, alpha=1e-3, beta=1, tv_weight=0):
    """
    Return the total loss

    :image: generated image
    :target_content: target image (content)
    :target_style: target image (style)
    :w_content: weighting factors (content)
    :w_style: weighting factors (style)
    :alpha: content coeff
    :beta: style coeff
    :tv_weight: total variation coeff
    """

    with slim.arg_scope(vgg.vgg_arg_scope()):
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            _, f_image = vgg.vgg_19(image, spatial_squeeze=False)
            _, f_target_content = vgg.vgg_19(target_content, spatial_squeeze=False)
            _, f_target_style = vgg.vgg_19(target_style, spatial_squeeze=False)

    f_image = filter_endpoints(f_image)
    f_target_content = filter_endpoints(f_target_content)
    f_target_style = filter_endpoints(f_target_style)

    content_loss = loss_content(f_image, f_target_content, w_content)
    style_loss = loss_style(f_image, f_target_style, w_style)
    tv_loss = total_variation_loss(image)
    loss = alpha*content_loss + beta*style_loss + tv_weight+tv_loss
    return content_loss, style_loss, tv_loss, loss
