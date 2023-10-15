def set_seed(seed=1):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
