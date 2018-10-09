import scipy
import numpy as np
import tensorflow as tf
from lib.loss import get_loss


vgg_mean = [103.939, 116.779, 123.68]


def load_image(path, size=None, max_size=None):
    """
    :path: path to the image
    :size: tuple
    :max_size: if not None, resize the image to a max height/width
    and keeping the ratio
    """
    img = scipy.misc.imread(path).astype(np.float)
    img = img[..., :3]
    h, w = img.shape[:2]

    if size is not None:
        return scipy.misc.imresize(img, size).astype(np.float)
    else:
        ratio = float(h)/w
        if max_size is not None and (np.max([h, w]) > max_size):
            if h > w:
                new_h, new_w = int(max_size), int(max_size/ratio)
            else:
                new_h, new_w = int(max_size*ratio), int(max_size)

            return scipy.misc.imresize(img, (new_h, new_w, 3)).astype(np.float)

        return img


def image_np_to_tf(image):
    """
    Convert an numpy image to a tensorflow object

    :image: numpy object
    """
    image = image.astype(np.float)
    # Substract VGG average value from ImageNet
    image[:, :, 0] -= vgg_mean[0]
    image[:, :, 1] -= vgg_mean[1]
    image[:, :, 2] -= vgg_mean[2]

    return tf.expand_dims(tf.constant(image, dtype=tf.float32), 0)


def generate_image(im_content, im_style, w_content, w_style,
                   alpha=1e-3, beta=50, tv_weight=0,
                   learning_rate=5, n_steps=1000,
                   p_ckpt='checkpoints/vgg_19.ckpt', verb=True,
                   im_init=None):
    """
    Return the generated image with the losses

    :im_content: target image (content), Tensorflow tensor
    :im_style: target image (style), Tensorflow tensor
    :w_content: weighting factors (content)
    :w_style: weighting factors (style)
    :alpha: content coeff
    :beta: style coeff
    :tv_weight: total variation coeff
    :learning_rate: learning rate
    :n_steps: max iterations
    :p_ckpt: path to checkpoints
    :verb: verbose
    :im_init: Initialization for the generated image (If none, use normal
    distribution)
    """

    if im_init is None:
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            new = tf.Variable(tf.random_normal(im_content.shape), trainable=True)
    else:
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            new = im_init

    run_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                intra_op_parallelism_threads=0,
                                gpu_options=tf.GPUOptions(
                                        force_gpu_compatible=True,
                                        allow_growth=True))

    c_loss, s_loss, tv_loss, loss = get_loss(new, im_content, im_style,
                                             w_content, w_style,
                                             alpha, beta, tv_weight)
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = adam.minimize(loss, var_list=new)
    losses = []
    c_losses = []
    s_losses = []
    tv_losses = []
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope='vgg'))
    best_loss = None
    best_img = None

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, p_ckpt)
        for i in xrange(n_steps):
            train_step.run()
            v_c_loss, v_s_loss, v_tv_loss, v_loss = sess.run(
                                            [c_loss, s_loss, tv_loss, loss])
            if verb:
                print('step=%d/%d, content_loss=%.2e, ' % (i+1, n_steps, v_c_loss)
                      + 'style_loss=%.2e, tv_loss=%.2e, ' % (v_s_loss, v_tv_loss)
                      + 'loss: %.2e' % (v_loss))

            if len(losses) == 0:
                best_loss = v_loss
                best_img = sess.run(new)[0]

            losses.append(v_loss)
            c_losses.append(v_c_loss)
            s_losses.append(v_s_loss)
            tv_losses.append(v_tv_loss)

            if best_loss > v_loss:
                best_loss = v_loss
                best_img = sess.run(new)[0]

    return best_img, losses, c_losses, s_losses, tv_losses
