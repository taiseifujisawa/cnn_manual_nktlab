def set_seed(seed=1):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(seed)
    tf.config.threading.set_intra_op_parallelism_threads(seed)
