#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

#use unbuffered python for output
dlrm_exe="python3 -u dlrm_s_pytorch.py"

use_rclone=false

dlrm_model_root=/models
dlrm_support_dir=/support

echo "[INFO] Launch Pytorch"

folder=$(date | sha1sum | head -c 6)
echo "[INFO] $folder is the new log location"

dlrm_log_dir="/avazu_redux_log/${folder}"
dlrm_model_dir="${dlrm_model_root}/${folder}"
mkdir -p $dlrm_log_dir
mkdir -p $dlrm_model_dir

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
--nepochs 1 --use-gpu ${randseed_arg}"

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
    $dlrm_exe --arch-mlp-bot ${1}${5} \
              --arch-sparse-feature-size $2 \
              --enable-linp \
              --enable-linp-up \
              --proj-down-dim $3 \
              --proj-up-dim $5 \
              --linp-init $4 \
              --save-model ${dlrm_model_dir}/linpup_${2}_${3}_${5}_${4}.m \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name drive:emb/$folder
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
    $dlrm_exe --arch-mlp-bot ${1}${2} \
              --arch-sparse-feature-size $2 \
              --enable-linp \
              --enable-linp-up \
              --proj-down-dim $3 \
              --linp-init $4 \
              --save-model ${dlrm_model_dir}/linp_${2}_${3}_${4}.m \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name drive:emb/$folder
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
    $dlrm_exe --arch-mlp-bot ${1}${2} \
              --arch-sparse-feature-size $2 \
              --enable-linp-up \
              --enable-rp \
              --rp-file $5 \
              --proj-down-dim $3 \
              ${proj_up_args} \
              --linp-init $4 \
              --save-model ${dlrm_model_dir}/rdlu_${2}_${3}_${4}.m \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name drive:emb/$folder
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
    $dlrm_exe --arch-mlp-bot ${1} \
              --arch-sparse-feature-size $2 \
              --enable-linp-up \
              --enable-rp \
              --rp-file $5 \
              --proj-down-dim $3 \
              ${proj_up_args} \
              --linp-init $4 \
              --save-model ${dlrm_model_dir}/rdlu_${2}_${3}_${4}.m \
              $generic_args 2>&1 | tee -a ${log_name}
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name drive:emb/$folder
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
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              --save-model ${dlrm_model_dir}/rp_${2}_${3}.m \
              $generic_args \
              $rp_args \
              --rp-file ${4} 2>&1 | tee -a $log_name
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name drive:emb/$folder
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
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              --save-model ${dlrm_model_dir}/rp_downup_${2}_${3}.m \
              $generic_args \
              $rp_args \
              --enable-rp-up \
              --rpup-file ${5} \
              --rp-file ${4} 2>&1 | tee -a $log_name
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name drive:emb/$folder
    fi
}



export CUDA_VISIBLE_DEVICES=0
# run_linp_up "1-256-128-" 64 64 normal

run_vanilla "1-256-128-128" 128
run_vanilla "1-256-128-64" 64
run_vanilla "1-256-128-32" 32
run_vanilla "1-256-128-16" 16
run_vanilla "1-256-128-8" 8
run_vanilla "1-256-128-4" 4

run_linp "1-256-128-" 64 4 normal
run_linp "1-256-128-" 64 8 normal
run_linp "1-256-128-" 64 16 normal
run_linp "1-256-128-" 64 32 normal
run_linp "1-256-128-" 64 64 normal

run_rp "1-256-128-4" 64 4 ${dlrm_support_dir}/rp_matrices/rpm_64_4_i0.bin
run_rp "1-256-128-8" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_i0.bin
run_rp "1-256-128-16" 64 16 ${dlrm_support_dir}/rp_matrices/rpm_64_16_i0.bin
run_rp "1-256-128-32" 64 32 ${dlrm_support_dir}/rp_matrices/rpm_64_32_i0.bin
run_rp "1-256-128-64" 64 64 ${dlrm_support_dir}/rp_matrices/rpm_64_64_i0.bin

#TP DownUp
# run_linp_up "1-256-128-" 64 4 normal
# run_linp_up "1-256-128-" 64 8 normal
# run_linp_up "1-256-128-" 64 16 normal
# run_linp_up "1-256-128-" 64 32 normal
# run_linp_up2 "1-256-128-" 64 4 normal 128

#RP DownUp
# run_rp_downup "1-256-128-64" 64 4 ${dlrm_support_dir}/rp_matrices/rpm_64_4_i0.bin ${dlrm_support_dir}/rp_matrices/rpm_4_64_i0.bin
# run_rp_downup "1-256-128-64" 64 8 ${dlrm_support_dir}/rp_matrices/rpm_64_8_i0.bin ${dlrm_support_dir}/rp_matrices/rpm_8_64_i0.bin
# run_rp_downup "1-256-128-64" 64 16 ${dlrm_support_dir}/rp_matrices/rpm_64_16_i0.bin ${dlrm_support_dir}/rp_matrices/rpm_16_64_i0.bin
# run_rp_downup "1-256-128-64" 64 32 ${dlrm_support_dir}/rp_matrices/rpm_64_32_i0.bin ${dlrm_support_dir}/rp_matrices/rpm_32_64_i0.bin

#RDLU
# run_rp_down_linp_up "1-256-128-" 64 4 normal ${dlrm_support_dir}/rp_matrices/rpm_64_4_i0.bin 
# run_rp_down_linp_up "1-256-128-" 64 8 normal ${dlrm_support_dir}/rp_matrices/rpm_64_8_i0.bin 
# run_rp_down_linp_up "1-256-128-" 64 16 normal ${dlrm_support_dir}/rp_matrices/rpm_64_16_i0.bin 
# run_rp_down_linp_up "1-256-128-" 64 32 normal ${dlrm_support_dir}/rp_matrices/rpm_64_32_i0.bin 
# run_rp_down_linp_up "1-256-128-" 64 64 normal ${dlrm_support_dir}/rp_matrices/rpm_64_64_i0.bin 

