#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

#use unbuffered python for output
dlrm_exe="python3 -u dlrm_s_pytorch.py"

use_rclone=false

dlrm_support_dir=/home/usr1/bghaem/mu/proj/dlrm/support

echo "[INFO] Launch Pytorch"

generic_args="--arch-mlp-top "512-256-1" --data-generation dataset
--data-set avazu
--avazu-db-path ${dlrm_support_dir}/data/avazu.db
--loss-function bce --round-targets True --learning-rate 0.2
--mini-batch-size 128 --num-workers 0
--print-freq 5000 --print-time --test-freq 30000 
--nepochs 1 --use-gpu"

rp_args="--enable-rp"

folder=$(date | sha1sum | head -c 6)
echo "[INFO] $folder is the new log location"

echo "[INFO] generic args: \n $generic_args"
dlrm_log_dir="/home/usr1/bghaem/mu/proj/dlrm/log/${folder}"

mkdir -p $dlrm_log_dir

# args: mlp-bot, sparse-feature-size
run_vanilla() {

    log_name="${dlrm_log_dir}/log_kaggle_dlrm_${2}.log"
    echo "[INFO] run dlrm --> mlp-bot: $1, sparse-feature-size: $2" \
          | tee $log_name
    echo "[INFO] generic args ${generic_args}"

    date | tee -a ${log_name}
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name gd:emb/$folder
    fi
}

# args: mlp-bot, sparse-feature-size-in, sparse-feature-size-out, rp_file_name
run_rp() {

    log_name="${dlrm_log_dir}/log_kaggle_rp_${2}_${3}.log"
    echo "[INFO] run rp --> mlp-bot: ${1}, sparse-feature-size-in: ${2}" \
          | tee $log_name
    echo "              --> sparse-feature-size-out: ${3}, rp_file_name ${4}" \
          | tee -a $log_name
    echo "[INFO] generic args ${generic_args}"

    date | tee -a $log_name
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              $generic_args \
              $rp_args \
              --rp-file ${4} 2>&1 | tee -a $log_name
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name gd:emb/$folder
    fi
}

#vanilla 64 
#run_vanilla "1-512-256-64-64" 64 !!(Too big for GTX2080)!!

#vanilla 32
#run_vanilla "1-512-256-64-32" 32 !!(Too big for GTX2080)!!



export CUDA_VISIBLE_DEVICES=0
#vanilla 32
run_vanilla "1-256-64-32" 32

#vanilla 16
run_vanilla "1-256-64-16" 16

#vanilla 8
run_vanilla "1-256-64-8" 8

#vanilla 4
run_vanilla "1-256-64-4" 4


export CUDA_VISIBLE_DEVICES=2 #requires too much gpu memory. Block gpu

#RP 64->4
run_rp "1-256-64-4" 64 4 ${dlrm_support_dir}/rp_matrices/rpm_64_4_i0.bin

#RP 64->8
run_rp "1-256-64-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_i0.bin

#RP 64->16
run_rp "1-256-64-16" 64 16 ${dlrm_support_dir}/rp_matrices/rpm_64_16_i0.bin

#RP 128->4
run_rp "1-256-64-4" 128 4 ${dlrm_support_dir}/rp_matrices/rpm_128_4_i0.bin

#RP 128->8
run_rp "1-256-64-8" 128 8 ${dlrm_support_dir}/rp_matrices/rpm_128_8_i0.bin

#RP 128->16
run_rp "1-256-64-16" 128 16 ${dlrm_support_dir}/rp_matrices/rpm_128_16_i0.bin

exit



