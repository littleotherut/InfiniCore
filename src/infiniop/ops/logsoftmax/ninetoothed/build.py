import ninetoothed
from ntops.kernels import log_softmax

import infiniop.ninetoothed.build

def build():
    MAX_NDIM = 5

    ndim_values = range(1, MAX_NDIM + 1)
    dtype_values = (
        ninetoothed.float16,
        ninetoothed.bfloat16,
        ninetoothed.float32,
        ninetoothed.float64,
    )
    
    for ndim in ndim_values:
        dim = ndim - 1  
        
        current_grid = {
            "ndim": (ndim,),
            "dtype": dtype_values,
            "dim": (dim,),  # 只生成最后一维
            "block_size": (1024,),
        }
        
        infiniop.ninetoothed.build.build(
            log_softmax.premake,
            current_grid,
            caller="cuda",
            op_name="logsoftmax",
            output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
        )
