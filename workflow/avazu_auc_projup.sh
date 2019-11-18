#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

#use unbuffered python for output
dlrm_exe="python3 -u dlrm_s_pytorch.py"

use_rclone=false

dlrm_model_root=/mnt/bcho/rp/models
dlrm_support_dir=/mnt/bcho/rp/support

echo "[INFO] Launch Pytorch"

folder=$(date | sha1sum | head -c 6)
echo "[INFO] $folder is the new log location"

dlrm_log_dir="/home/bcho/rp/avazu_redux_log/${folder}"
dlrm_model_dir="${dlrm_model_root}"
mkdir -p $dlrm_log_dir

randseed_arg=""
if [ -n "$1" ] ; then
  randseed_arg="--numpy-rand-seed $1"
fi

generic_args="--arch-mlp-top "512-256-1" --data-generation dataset
--data-set avazu
--avazu-db-path ${dlrm_support_dir}/data/avazu.db
--loss-function bce --round-targets True --learning-rate 0.2
--mini-batch-size 128 --num-workers 0
--print-freq 5000 --print-time --test-freq 30000
--nepochs 1 ${randseed_arg}"

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
    model_dir=${dlrm_model_dir}/${3}
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              --load-model ${model_dir}/dlrm_${2}.m \
              --auc-only \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name utdrive:emb/$folder
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
    model_dir=${dlrm_model_dir}/${5}
    $dlrm_exe --arch-mlp-bot ${1}${3} \
              --arch-sparse-feature-size $2 \
              --enable-linp \
              --linp-init $4 \
              --load-model ${model_dir}/linp_${2}_${3}_${4}.m \
              --auc-only \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name utdrive:emb/$folder
    fi
}

run_linp_up2() {
    log_name="${dlrm_log_dir}/log_linpup_${2}_${3}_${5}_${4}.log"
    echo "[INFO] run linp --> mlp-bot: ${1}${3}, sparse-feature-size-out: ${2}" \
          | tee $log_name
    echo "                --> sparse-feature-size-out: ${3}, init_fn ${4}" \
          | tee -a $log_name
    echo "                --> projections: ${2} -> ${3} -> ${5}" \
          | tee -a $log_name
    echo "[INFO] generic args ${generic_args}" \
          | tee -a $log_name

    date | tee -a ${log_name}
    model_dir=${dlrm_model_dir}/${6}
    $dlrm_exe --arch-mlp-bot ${1}${5} \
              --arch-sparse-feature-size $2 \
              --enable-linp \
              --enable-linp-up \
              --proj-down-dim $3 \
              --proj-up-dim $5 \
              --linp-init $4 \
              --load-model ${model_dir}/linpup_${2}_${3}_${5}_${4}.m \
              --auc-only \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name utdrive:emb/$folder
    fi
}
run_linp_up() {
    log_name="${dlrm_log_dir}/log_linp_${2}_${3}_${4}.log"
    echo "[INFO] run linp --> mlp-bot: ${1}${3}, sparse-feature-size-out: ${2}" \
          | tee $log_name
    echo "                --> sparse-feature-size-out: ${3}, init_fn ${4}" \
          | tee -a $log_name
    echo "[INFO] generic args ${generic_args}" \
          | tee -a $log_name

    date | tee -a ${log_name}
    model_dir=${dlrm_model_dir}/${5}
    $dlrm_exe --arch-mlp-bot ${1}${2} \
              --arch-sparse-feature-size $2 \
              --enable-linp \
              --enable-linp-up \
              --proj-down-dim $3 \
              --linp-init $4 \
              --load-model ${model_dir}/linp_${2}_${3}_${4}.m \
              --auc-only \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name utdrive:emb/$folder
    fi
}

run_rp_down_linp_up() {
    log_name="${dlrm_log_dir}/log_rdlu_${2}_${3}_${4}.log"
    echo "[INFO] run linp --> mlp-bot: ${1}${3}, sparse-feature-size-out: ${2}" \
          | tee $log_name
    echo "                --> sparse-feature-size-out: ${3}, init_fn ${4}" \
          | tee -a $log_name
    echo "[INFO] generic args ${generic_args}" \
          | tee -a $log_name

    date | tee -a ${log_name}
    if [ -n "$6" ] ; then
      proj_up_args="--proj-up-dim $6"
      echo "proj_up_args = ${proj_up_args}"
    fi
    model_dir=${dlrm_model_dir}/${7}
    $dlrm_exe --arch-mlp-bot ${1}${2} \
              --arch-sparse-feature-size $2 \
              --enable-linp-up \
              --enable-rp \
              --rp-file $5 \
              --proj-down-dim $3 \
              ${proj_up_args} \
              --linp-init $4 \
              --load-model ${model_dir}/rdlu_${2}_${3}_${4}.m \
              --auc-only \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name utdrive:emb/$folder
    fi
}

run_rp_down_linp_up2() {
    log_name="${dlrm_log_dir}/log_rdlu_${2}_${3}_${6}_${4}.log"
    echo "[INFO] run linp --> mlp-bot: ${1}${3}, sparse-feature-size-out: ${2}" \
          | tee $log_name
    echo "                --> sparse-feature-size-out: ${3}, init_fn ${4}" \
          | tee -a $log_name
    echo "[INFO] generic args ${generic_args}" \
          | tee -a $log_name

    date | tee -a ${log_name}
    if [ -n "$6" ] ; then
      proj_up_args="--proj-up-dim $6"
      echo "proj_up_args = ${proj_up_args}"
    fi
    model_dir=${dlrm_model_dir}/${7}
    $dlrm_exe --arch-mlp-bot ${1} \
              --arch-sparse-feature-size $2 \
              --enable-linp-up \
              --enable-rp \
              --rp-file $5 \
              --proj-down-dim $3 \
              ${proj_up_args} \
              --linp-init $4 \
              --load-model ${model_dir}/rdlu_${2}_${3}_${4}.m \
              --auc-only \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name utdrive:emb/$folder
    fi
}


# args: mlp-bot, sparse-feature-size-in, sparse-feature-size-out, rp_file_name
run_rp() {

    log_name="${dlrm_log_dir}/log_rp_${2}_${3}.log"
    echo "[INFO] run rp --> mlp-bot: ${1}, sparse-feature-size-in: ${2}" \
          | tee $log_name
    echo "              --> sparse-feature-size-out: ${3}, rp_file_name ${4}" \
          | tee -a $log_name
    echo "[INFO] generic args ${generic_args}" \
          | tee -a $log_name

    date | tee -a $log_name
    model_dir=${dlrm_model_dir}/${5}
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              --load-model ${model_dir}/rp_${2}_${3}.m \
              --auc-only \
              $generic_args \
              $rp_args \
              --rp-file ${4} 2>&1 | tee -a $log_name
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name utdrive:emb/$folder
    fi
}

run_rp_downup() {

    log_name="${dlrm_log_dir}/log_rp_${2}_${3}.log"
    echo "[INFO] run rp --> mlp-bot: ${1}, sparse-feature-size-in: ${2}" \
          | tee $log_name
    echo "              --> sparse-feature-size-out: (${3})->${2}, rp_file_name ${4}, rpup_file_name ${5}" \
          | tee -a $log_name
    echo "[INFO] generic args ${generic_args}" \
          | tee -a $log_name

    date | tee -a $log_name
    model_dir=${dlrm_model_dir}/${6}
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              --load-model ${model_dir}/rp_downup_${2}_${3}.m \
              --auc-only \
              $generic_args \
              $rp_args \
              --enable-rp-up \
              --rpup-file ${5} \
              --rp-file ${4} 2>&1 | tee -a $log_name
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name utdrive:emb/$folder
    fi
}



# export CUDA_VISIBLE_DEVICES=0

# run_vanilla "1-256-128-128" 128 f36bfb
# run_vanilla "1-256-128-64" 64 f36bfb
# run_vanilla "1-256-128-32" 32 f36bfb
# run_vanilla "1-256-128-16" 16 f36bfb
# run_vanilla "1-256-128-8" 8 f36bfb
# run_vanilla "1-256-128-4" 4 f36bfb

# run_linp "1-256-128-" 64 4 normal f36bfb
# run_linp "1-256-128-" 64 8 normal f36bfb
# run_linp "1-256-128-" 64 16 normal f36bfb
# run_linp "1-256-128-" 64 32 normal f36bfb
# run_linp "1-256-128-" 64 64 normal f36bfb

# run_rp "1-256-128-4" 64 4 ${dlrm_support_dir}/rp_matrices/rpm_64_4_i0.bin f36bfb
# run_rp "1-256-128-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_i0.bin f36bfb
# run_rp "1-256-128-16" 64 16 ${dlrm_support_dir}/rp_matrices/rpm_64_16_i0.bin f36bfb
# run_rp "1-256-128-32" 64 32 ${dlrm_support_dir}/rp_matrices/rpm_64_32_i0.bin f36bfb
# run_rp "1-256-128-64" 64 64 ${dlrm_support_dir}/rp_matrices/rpm_64_64_i0.bin f36bfb

#TP DownUp
# run_linp_up "1-256-128-" 64 4 normal 2f940e
# run_linp_up "1-256-128-" 64 8 normal ea9087
# run_linp_up "1-256-128-" 64 16 normal ea9087
# run_linp_up "1-256-128-" 64 32 normal ea9087
# run_linp_up "1-256-128-" 64 64 normal 711cb7
# run_linp_up2 "1-256-128-" 64 4 normal 128 f9553a

#RP DownUp
# run_rp_downup "1-256-64" 64 4 ${dlrm_support_dir}/rp_matrices/rpm_64_4_i0.bin ${dlrm_support_dir}/rp_matrices/rpm_4_64_i0.bin 1bdc5c
# run_rp_downup "1-256-64" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_i0.bin ${dlrm_support_dir}/rp_matrices/rpm_8_64_i0.bin 1bdc5c
# run_rp_downup "1-256-64" 64 16 ${dlrm_support_dir}/rp_matrices/rpm_64_16_i0.bin ${dlrm_support_dir}/rp_matrices/rpm_16_64_i0.bin 1bdc5c
# run_rp_downup "1-256-64" 64 32 ${dlrm_support_dir}/rp_matrices/rpm_64_32_i0.bin ${dlrm_support_dir}/rp_matrices/rpm_32_64_i0.bin 84cea7

#RDLU
# run_rp_down_linp_up "1-256-128-" 64 4 normal ${dlrm_support_dir}/rp_matrices/rpm_64_4_i0.bin 64 f47e4b
# run_rp_down_linp_up "1-256-128-" 64 8 normal ${dlrm_support_dir}/rp_matrices/rpm_64_8_i0.bin 64 711cb7
# run_rp_down_linp_up "1-256-128-" 64 16 normal ${dlrm_support_dir}/rp_matrices/rpm_64_16_i0.bin 64 711cb7
# run_rp_down_linp_up "1-256-128-" 64 32 normal ${dlrm_support_dir}/rp_matrices/rpm_64_32_i0.bin 64 711cb7
# run_rp_down_linp_up "1-256-128-" 64 64 normal ${dlrm_support_dir}/rp_matrices/rpm_64_64_i0.bin 64 834fb5


