srun --partition=mt --nodes=1 --gres=gpu:mt:1 --ntasks=1 --cpus-per-task=16 --mem=256G --time=00:03:00  \
        --output=T1-1-28_moore%j.log          \
        python test/infinicore/run.py --moore --eq_nan --bench --ops bitwise_left_shift --debug
