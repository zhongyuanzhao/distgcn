import tensorflow as tf

def my_softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.nn.weighted_cross_entropy_with_logits(logits=preds, targets=labels, pos_weight=0.1)
    return tf.reduce_mean(loss)


def my_weighted_softmax_cross_entropy(preds, labels, weights):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.nn.weighted_cross_entropy_with_logits(logits=preds, targets=labels, pos_weight=0.1)
    # norm_wts = tf.norm(weights, axis=0)
    norm_wts = weights/tf.reduce_mean(weights)
    return tf.reduce_mean(tf.math.multiply(loss, norm_wts))


def my_accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def my_f1(preds, labels):
    """Precision with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    labels_pos = tf.cast(labels[:,1], tf.bool)
    labels_neg = tf.cast(labels[:,0], tf.bool)
    wrong_prediction = tf.logical_not(correct_prediction)
    tp = tf.cast(tf.logical_and(correct_prediction, labels_pos), tf.float32)
    # tn = tf.cast(tf.logical_and(correct_prediction, labels_neg), tf.float32)
    fp = tf.cast(tf.logical_and(wrong_prediction, labels_neg), tf.float32)
    fn = tf.cast(tf.logical_and(wrong_prediction, labels_pos), tf.float32)
    tp_all = tf.reduce_mean(tp)
    # tn_all = tf.reduce_mean(tn)
    fp_all = tf.reduce_mean(fp)
    fn_all = tf.reduce_mean(fn)
    precision = tp_all/(tp_all+fp_all)
    recall = tp_all/(tp_all+fn_all)
    f1 = 2 * (precision * recall)/(precision+recall)
    return f1, precision, recall


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
