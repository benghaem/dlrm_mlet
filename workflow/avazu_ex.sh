#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

#use unbuffered python for output
dlrm_exe="python3 -u dlrm_s_pytorch.py"

use_rclone=true

dlrm_model_root=/dlrm/models
dlrm_support_dir=/dlrm/support

echo "[INFO] Launch Pytorch"

folder=$(date | sha1sum | head -c 6)
echo "[INFO] $folder is the new log location"

dlrm_log_dir="/dlrm/avazu_redux_log/${folder}"
dlrm_model_dir="${dlrm_model_root}/${folder}"
mkdir -p $dlrm_log_dir
mkdir -p $dlrm_model_dir


generic_args="--arch-mlp-top "512-256-1" --data-generation dataset
--data-set avazu
--avazu-db-path ${dlrm_support_dir}/data/avazu.db
--loss-function bce --round-targets True --learning-rate 0.2
--mini-batch-size 128 --num-workers 0
--print-freq 5000 --print-time --test-freq 30000
--nepochs 1 --use-gpu --numpy-rand-seed 882"

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

# args: mlp-bot-partial, sparse-feature-size-in, sparse-feature-size-out, init_fn
run_linp() {

    log_name="${dlrm_log_dir}/log_linp_${2}_${3}_${4}.log"
    echo "[INFO] run linp --> mlp-bot: ${1}${3}, sparse-feature-size-out: ${2}" \
          | tee $log_name
    echo "                --> sparse-feature-size-out: ${3}, init_fn ${4}" \
          | tee -a $log_name
    echo "[INFO] generic args ${generic_args}" \
          | tee -a $log_name

    date | tee -a ${log_name}
    $dlrm_exe --arch-mlp-bot ${1}${3} \
              --arch-sparse-feature-size $2 \
              --enable-linp \
              --linp-init $4 \
              --save-model ${dlrm_model_dir}/linp_${2}_${3}_${4}.m \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name drive:emb/$folder
    fi
}

# args: mlp-bot, sparse-feature-size-in, sparse-feature-size-out, rp_file_name,
# extra
run_rp() {

    log_name="${dlrm_log_dir}/log_rp_${2}_${3}_${5}.log"
    echo "[INFO] run rp --> mlp-bot: ${1}, sparse-feature-size-in: ${2}" \
          | tee $log_name
    echo "              --> sparse-feature-size-out: ${3}, rp_file_name ${4}" \
          | tee -a $log_name
    echo "[INFO] generic args ${generic_args}" \
          | tee -a $log_name

    date | tee -a $log_name
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              --save-model ${dlrm_model_dir}/rp_${2}_${3}_${5}.m \
              $generic_args \
              $rp_args \
              --rp-file ${4} 2>&1 | tee -a $log_name
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name drive:emb/$folder
    fi
}

#vanilla 4


#run_linp "1-256-128-" 64 4 normal
#
#run_linp "1-256-128-" 64 8 normal
#
#run_linp "1-256-128-" 64 16 normal



#RP 64->4
export CUDA_VISIBLE_DEVICES=0
run_rp "1-256-128-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_mu_i1.bin 882_1
run_rp "1-256-128-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_mu_i2.bin 882_2
run_rp "1-256-128-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_mu_i3.bin 882_3
run_rp "1-256-128-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_mu_i4.bin 882_4
run_rp "1-256-128-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_mu_i5.bin 882_5
run_rp "1-256-128-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_mu_i6.bin 882_6
run_rp "1-256-128-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_mu_i7.bin 882_7


exit


#vanilla 32
#run_vanilla "1-512-256-64-32" 32 !!(Too big for GTX2080)!!

export CUDA_VISIBLE_DEVICES=1

run_linp "1-256-64-" 64 4 normal

run_linp "1-256-64-" 64 8 normal

run_linp "1-256-64-" 64 16 normal

exit

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



