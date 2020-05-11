import tensorflow as tf

def decode_libsvm(line):
    # columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
    # features = dict(zip(CSV_COLUMNS, columns))
    # labels = features.pop(LABEL_COLUMN)
    columns = line.split()  #tf.string_split([line], ' ')
    labels = columns[0]  #tf.string_to_number(columns.values[0], out_type=tf.float32)
    splits = [e.split(':') for e in columns[1:]] #tf.string_split(columns.values[1:], ':')
    #id_vals = tf.reshape(splits.values, splits.dense_shape)
    feat_ids = [e[0] for e in splits]  #tf.split(id_vals, num_or_size_splits=2, axis=1)
    feat_vals = [e[1] for e in splits]
    #feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
    #feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
    # feat_ids = tf.reshape(feat_ids,shape=[-1,FLAGS.field_size])
    # for i in range(splits.dense_shape.eval()[0]):
    #    feat_ids.append(tf.string_to_number(splits.values[2*i], out_type=tf.int32))
    #    feat_vals.append(tf.string_to_number(splits.values[2*i+1]))
    # return tf.reshape(feat_ids,shape=[-1,field_size]), tf.reshape(feat_vals,shape=[-1,field_size]), labels
    return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

if __name__ == "__main__":
    with open('./data/train0.txt', 'r') as f:
        for line in f:
            res = decode_libsvm(line)
            a=1

    #v = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; x = [7, 8]
    v = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]); x = tf.constant([7, 8], shape=[2, 1])
    sess = tf.Session()
    embeddings = tf.multiply(v, x)
    sum_square = tf.square(tf.reduce_sum(embeddings, 1))
    square_sum = tf.reduce_sum(tf.square(embeddings), 1)
    aa = sess.run(embeddings)
    pass

    r = 0.88389314996;  p = 0.585340222436; f = 2 / (1 / r + 1 / p)
    pass