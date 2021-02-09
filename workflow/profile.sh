#use unbuffered python for output
dlrm_exe="nvprof python3 -u dlrm_s_pytorch.py"


dlrm_model_root=/dlrm/models
dlrm_support_dir=/dlrm/support

echo "[INFO] Launch Pytorch"

folder=$(date | sha1sum | head -c 6)
echo "[INFO] $folder is the new log location"

dlrm_log_dir="/dlrm/avazu_redux_log/profile/${folder}"
dlrm_model_dir="${dlrm_model_root}/${folder}"
mkdir -p $dlrm_log_dir
mkdir -p $dlrm_model_dir


generic_args="--arch-mlp-top "512-256-1" --data-generation dataset
--data-set avazu
--avazu-db-path ${dlrm_support_dir}/data/avazu.db
--loss-function bce --round-targets True --learning-rate 0.2
--mini-batch-size 128 --num-workers 1
--print-freq 5000 --print-time --test-freq 30000
--nepochs 1 --use-gpu"

rp_args="--enable-rp"
echo "[INFO] generic args: \n $generic_args"


# args: mlp-bot, sparse-feature-size
run_vanilla() {

    log_name="${dlrm_log_dir}/log_dlrm_${2}.log"
    echo "[INFO] run dlrm --> mlp-bot: $1, sparse-feature-size: $2" \
          | tee $log_name
    echo "[INFO] generic args ${generic_args}" \
          | tee -a ${log_name}

    date | tee -a ${log_name}
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              --save-model ${dlrm_model_dir}/dlrm_${2}.m \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name drive:emb/$folder
    fi
}

run_vanilla "1-256-128-4" 4
