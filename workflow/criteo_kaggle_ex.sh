#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

dlrm_exe="python3 -u dlrm_s_pytorch.py"

use_rclone=false

dlrm_support_dir=/home/usr1/bghaem/mu/proj/dlrm/support

echo "[INFO] Launch Pytorch"

generic_args="--arch-mlp-top "512-256-1" --data-generation dataset
--data-set kaggle
--processed-data-file ${dlrm_support_dir}/data/kaggle_processed.npz
--loss-function bce --round-targets True --learning-rate 0.2
--mini-batch-size 128 --num-workers 0
--print-freq 128 --print-time --test-freq 3000 
--nepochs 1 --use-gpu"

rp_args="--enable-rp"

folder=$(date | sha1sum | head -c 6)
echo "[INFO] $folder is the new log location"

log_dir=/tmp/kaggle_logs/${folder}
mkdir -p $log_dir

echo "[INFO] generic args: $generic_args"

# args: mlp-bot, sparse-feature-size
run_vanilla() {

    log_name="${log_dir}/log_kaggle_dlrm_${2}.log" echo "[INFO] run dlrm --> mlp-bot: $1, sparse-feature-size: $2" \
          | tee $log_name

    date | tee -a $log_name
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              $generic_args \
              2>&1 | tee -a $log_name
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name gd:emb/$folder
    fi
}

# args: mlp-bot, sparse-feature-size-in, sparse-feature-size-out, rp_file_name
run_rp() {

    log_name="${log_dir}/log_kaggle_rp_${2}_${3}.log"
    echo "[INFO] run rp --> mlp-bot: ${1}, sparse-feature-size-in: ${2}" \
          | tee $log_name
    echo "              --> sparse-feature-size-out: ${3}, rp_file_name ${4}" \
          | tee -a $log_name

    date | tee -a $log_name
    $dlrm_exe --arch-mlp-bot $1 \
              --arch-sparse-feature-size $2 \
              $generic_args \
              $rp_args \
              --rp-file ${4} \
              2>&1 | tee -a $log_name
    date | tee -a $log_name
    if [ "$use_rclone" = true ] ; then
        rclone copy $log_name gd:emb/$folder
    fi
}

#RP 128->32
#run_rp "13-512-256-64-32" 128 32 ../rp_matrices/rpm_128_32_i0.bin

#RP 128->8
#run_rp "13-512-256-64-8" 128 8 ../rp_matrices/rpm_128_8_i0.bin

#RP 64->4
# run_rp "13-512-256-64-4" 64 4 ${dlrm_support_dir}/rp_matrices/rpm_64_4_i0.bin

#DLRM 4

export CUDA_VISIBLE_DEVICES=0
run_vanilla "13-512-256-64-4" 4 
exit

export CUDA_VISIBLE_DEVICES=1
run_vanilla "13-512-256-64-8" 8 &

wait
wait

#RP 64->8
#run_rp "13-512-256-64-8" 64 8 ../rp_matrices/rpm_64_8_i0.bin

#RP 64->16
#run_rp "13-512-256-64-16" 64 16 ../rp_matrices/rpm_64_16_i0.bin

exit



#
#mat_file=../rp_matrices/matrices_128_8.bin
#echo "RP 128->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_128_8.log
#date
#rclone copy log_kaggle_rp_128_8.log gd:emb/$folder

#echo "DLRM 8"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 8 --arch-mlp-bot "13-512-256-64-8" $generic_args  2>&1 | tee log_kaggle_dlrm8.log
#date
#rclone copy log_kaggle_dlrm8.log gd:emb/$folder
#
#echo "DLRM 32"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 32 --arch-mlp-bot "13-512-256-64-32" $generic_args  2>&1 | tee log_kaggle_dlrm32.log
#date
#rclone copy log_kaggle_dlrm32.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_64_8.bin
#echo "RP 64->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_8.log
#date
#rclone copy log_kaggle_rp_64_8.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_32_8.bin
#echo "RP 32->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 32 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_32_8.log
#date
#rclone copy log_kaggle_rp_32_8.log gd:emb/$folder


#mat_file=../rp_matrices/matrices_32_4.bin
#echo "RP 32->4"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 32 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_32_4.log
#date
#rclone copy log_kaggle_rp_32_4.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_32_16.bin
#echo "RP 32->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 32 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_32_16.log
#date
#rclone copy log_kaggle_rp_32_16.log gd:emb/$folder


#echo "DLRM 16"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 16 --arch-mlp-bot "13-512-256-64-16" $generic_args  2>&1 | tee log_kaggle_dlrm16.log
#date
#rclone copy log_kaggle_dlrm16.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_64_16.bin
#echo "RP 64->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_16.log
#date
#rclone copy log_kaggle_rp_64_16.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_64_64.bin
#echo "RP 64->64"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-64" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_64.log
#date
#rclone copy log_kaggle_rp_64_64.log gd:emb/$folder



#mat_file=../rp_matrices/matrices_64_16.bin
#echo "RP 64->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_16.log
#date
#rclone copy log_kaggle_rp_64_16.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_64_64.bin
#echo "RP 64->64"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-64" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_64.log
#date
#rclone copy log_kaggle_rp_64_64.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_128_4_i1.bin
#echo "RP 128->4"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_128_4_i1.log
#date
#rclone copy log_kaggle_rp_128_4_i1.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_64_4_i1.bin
#echo "RP 64->4"
#echo $mat_file
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_4_i1.log
#date
#rclone copy log_kaggle_rp_64_4_i1.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_128_4_i2.bin
#echo "RP 128->4"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_128_4_i2.log
#date
#rclone copy log_kaggle_rp_128_4_i2.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_64_4_i2.bin
#echo "RP 64->4"
#echo $mat_file
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_4_i2.log
#date
#rclone copy log_kaggle_rp_64_4_i2.log gd:emb/$folder


#mat_file=../rp_matrices/matrices_64_16_i1.bin
#echo "RP 64->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_16_i1.log
#date
#rclone copy log_kaggle_rp_64_16_i1.log gd:emb/$folder

##mat_file=../rp_matrices/matrices_64_16_i2.bin
#echo "RP 64->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_16_i2.log
#date
#rclone copy log_kaggle_rp_64_16_i2.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_64_16_i3.bin
#echo "RP 64->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_16_i3.log
#date
#rclone copy log_kaggle_rp_64_16_i3.log gd:emb/$folder




#mat_file=../rp_matrices/matrices_32_16_i1.bin
#echo "RP 32->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 32 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_32_16_i1.log
#date
#rclone copy log_kaggle_rp_32_16_i1.log gd:emb/$folder


#mat_file=../rp_matrices/matrices_16_8_i0.bin
#echo "RP 16->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 16 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_16_8_i0.log
#date
#rclone copy log_kaggle_rp_16_8_i0.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_16_8_i1.bin
#echo "RP 16->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 16 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_16_8_i1.log
#date
#rclone copy log_kaggle_rp_16_8_i1.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_128_8_i1.bin
#echo "RP 128->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_128_8_i1.log
#date
#rclone copy log_kaggle_rp_128_8_i1.log gd:emb/$folder
#
#
#mat_file=../rp_matrices/matrices_64_8_i2.bin
#echo "RP 64->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_8_i2.log
#date
#rclone copy log_kaggle_rp_64_8_i2.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_32_8_i1.bin
#echo "RP 32->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 32 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_32_8_i1.log
#date
#rclone copy log_kaggle_rp_32_8_i1.log gd:emb/$folder


#mat_file=../rp_matrices/matrices_16_4_i1.bin
#echo "RP 16->4"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 16 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_16_4_i1.log
#date
#rclone copy log_kaggle_rp_16_4_i1.log gd:emb/$folder
#
#
#mat_file=../rp_matrices/matrices_16_8_i2.bin
#echo "RP 16->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 16 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_16_8_i2.log
#date
#rclone copy log_kaggle_rp_16_8_i2.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_32_8_i2.bin
#echo "RP 32->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 32 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_32_8_i2.log
#date
#rclone copy log_kaggle_rp_32_8_i2.log gd:emb/$folder


#mat_file=../rp_matrices/matrices_64_16_spnz_8_i0.bin
#log_file=log_kaggle_rp_sp_64_16_spnz_8_i0.log
#echo "RP SP 64->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file 2>&1 | tee $log_file
#date
#rclone copy $log_file gd:emb/$folder
#
#
#mat_file=../rp_matrices/matrices_64_8_spnz_4_i2.bin
#log_file=log_kaggle_rp_sp_64_8_spnz_4_i2.log
#echo "RP SP 64->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file 2>&1 | tee $log_file
#date
#rclone copy $log_file gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_64_4_spnz_2_i0.bin
#log_file=log_kaggle_rp_sp_64_4_spnz_2_i0.log
#echo "RP SP 64->4"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file 2>&1 | tee $log_file
#date
#rclone copy $log_file gd:emb/$folder
#
#exit
#
#
#mat_file=../rp_matrices/matrices_128_4_spnz_2_i0.bin
#echo "RPSP 128->4"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rpsp_128_4_2_i0.log
#date
#rclone copy log_kaggle_rpsp_128_4_2_i0.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_128_8_spnz_4_i0.bin
#echo "RPSP 128->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rpsp_128_8_4_i0.log
#date
#rclone copy log_kaggle_rpsp_128_8_4_i0.log gd:emb/$folder
#
#mat_file=../rp_matrices/matrices_128_16_spnz_8_i0.bin
#echo "RPSP 128->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rpsp_128_16_8_i0.log
#date
#rclone copy log_kaggle_rpsp_128_16_8_i0.log gd:emb/$folder
#
#exit


#mat_file=../rp_matrices/matrices_128_4_i3.bin
#echo "RP 128->4"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_128_4_i3.log
#date
#rclone copy log_kaggle_rp_128_4_i3.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_64_4_i3.bin
#echo "RP 64->4"
#echo $mat_file
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_4_i3.log
#date
#rclone copy log_kaggle_rp_64_4_i3.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_64_8_i1.bin
#echo "RP 64->8"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-8" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_64_8_i1.log
#date
#rclone copy log_kaggle_rp_64_8_i1.log gd:emb/$folder

#
#
#mat_file=../rp_matrices/matrices_128_16_i0.bin
#echo "RP 128->16"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_128_16_i0.log
#date
#rclone copy log_kaggle_rp_128_16_i0.log gd:emb/$folder
#
#exit

#force 1 gpu
#export CUDA_VISIBLE_DEVICES=1
#echo "LIN 64->16 3epoch"
#date
#$dlrm_pt_bin --save-onnx --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-16" $generic_args $lin_args 2>&1 | tee log_kaggle_lin_64_16_3epoch.log
#date
#rclone copy log_kaggle_lin_64_16_3epoch.log gd:emb/$folder


#export CUDA_VISIBLE_DEVICES=0,1

#echo "DLRM 128"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 128 --arch-mlp-bot "13-512-256-128-128" ${generic_args}  2>&1 | tee log_kaggle_dlrm128.log
#date
#rclone copy log_kaggle_dlrm128.log gd:emb/$folder

#echo "DLRM 96"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 96 --arch-mlp-bot "13-512-256-128-96" ${generic_args}  2>&1 | tee log_kaggle_dlrm96.log
#date
#rclone copy log_kaggle_dlrm96.log gd:emb/$folder
#
#echo "DLRM 256"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 256 --arch-mlp-bot "13-512-256-256" ${generic_args}  2>&1 | tee log_kaggle_dlrm256.log
#date
#rclone copy log_kaggle_dlrm256.log gd:emb/$folder

#exit 
#echo "DLRM 64"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 64 --arch-mlp-bot "13-512-256-64-64" ${generic_args}  2>&1 | tee log_kaggle_dlrm64.log
#date
#rclone copy log_kaggle_dlrm64.log gd:emb/$folder

#echo "DLRM 4"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 4 --arch-mlp-bot "13-512-256-64-4" $generic_args  2>&1 | tee log_kaggle_dlrm4.log
#date
#rclone copy log_kaggle_dlrm4.log gd:emb/$folder

#mat_file=../rp_matrices/matrices_16_4.bin
#echo "RP 16->4"
#echo $mat_file
#date
#$dlrm_pt_bin --arch-sparse-feature-size 16 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices-file $mat_file  2>&1 | tee log_kaggle_rp_16_4.log
#date
#rclone copy log_kaggle_rp_16_4.log gd:emb/$folder
#
##early exit
#exit
#
#mat_file=../rp_matrices/mat_16_64_spnz_8.bin
#echo "RP SP 64->16"
#echo "mat_file"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 16 --arch-mlp-bot "13-512-256-64-16" $generic_args $rp_args --rp-matrices--file $mat_file --save-model rp_sp_64_16_spnz_8.model 2>&1 | tee log_kaggle_rp_sp_64_16_spnz_8.log
#date
#rclone copy log_kaggle_rp_sp_64_16_spnz_8.log gd:emb/$folder
#
#mat_file=../rp_matrices/mat_4_64_spnz_2.bin
#echo "RP SP 64->4"
#echo "mat_file"
#date
#$dlrm_pt_bin --arch-sparse-feature-size 4 --arch-mlp-bot "13-512-256-64-4" $generic_args $rp_args --rp-matrices--file $mat_file --save-model rp_sp_64_4_spnz_2.model 2>&1 | tee log_kaggle_rp_sp_64_4_spnz_2.log
#date
#rclone copy log_kaggle_rp_sp_64_4_spnz_2.log gd:emb/$folder


