#SBATCH -J bg_rp                # Job name
#SBATCH -o bg_rp.o%j            # Name of stdout output file
#SBATCH -e bg_rp.e%j            # Name of stderr error file
#SBATCH -p v100                 # Queue (partition) name
#SBATCH -N 1                    # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                    # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 04:00:00             # Run time (hh:mm:ss)
#SBATCH -A Accelerating-DNN-Tra # Project/Allocation

#SBATCH --mail-type=all
#SBATCH --mail-user=ben.ghaem@utexas.edu

#Let's assume jobs are in the 4hr range?
#p100s have FP16 support
#v100s have FP16 support and FP16 tensor cores <-- try to get v100s
# See 
# https://docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html#hardware-precision-matrix
# for more info

#Copy over data to /tmp 

cp -r $work/dlrm_support /tmp

#Copy over scripts to /tmp

cp -r $work/dlrm_rp /tmp

#Run the experiment scripts

/tmp/dlrm_rp/workflow/experiment.sh

#Save the log files

cp -r logs/* $work/dlrm_logs

#Cleanup

rm -rf /tmp/dlrm_rp
rm -rf /tmp/dlrm_support

