# Replication Script

import subprocess
from subprocess import PIPE
import datetime
import hashlib
import platform
import sys
import random
import time


dlrm_exe = ["python3", "-u", "dlrm_s_pytorch.py"]
rpmb_exe = ["python3", "rpmb.py"]

dlrm_support_dir = "/home/usr1/bghaem/mu/proj/dlrm/support"
dlrm_output_dir = "/home/usr1/bghaem/mu/proj/dlrm/paper_rep"
rp_mat_dir = "/home/usr1/bghaem/mu/proj/dlrm/paper_rep/rp_mats"

generic_args = {
    "arch-mlp-top": "512-256-1",
    "data-generation": "dataset",
    "loss-function": "bce",
    "round-targets": "True",
    "learning-rate": 0.2,
    "mini-batch-size": 128,
    "num-workers": 0,
    "print-freq": 5000,
    "print-time": None,
    "test-freq": 30000,
    "nepochs": 1,
    "use-gpu": None,
}

avazu_args = {"data-set": "avazu", "avazu-db-path": dlrm_support_dir + "/data/avazu.db"}

criteo_kaggle_args = {
             "data-set": "kaggle",
              "processed-data-file": dlrm_support_dir + "/data/kaggle_processed.npz",
        }

test_run_args = {"num-batches": 1000}


def gen_rp_mat(size_in, size_out):

    rp_path = None
    rp_file_id = None
    not_yet_generated = True
    while not_yet_generated:
        rp_file_id = hashlib.sha256(
            str(datetime.datetime.now()).encode("ascii")
        ).hexdigest()[0:6]

        rp_folder = rp_mat_dir
        rp_file_name = "ag_rpm_{}_{}_{}_{}.bin".format(
            size_in, size_out, platform.node(), rp_file_id
        )

        rp_path = rp_folder + "/" + rp_file_name

        # rpmb input is rows, cols so we need to do out, in
        rpmb_args = [str(size_out), str(size_in), "0", "dense", "-o", rp_path]
        p = subprocess.run(rpmb_exe + rpmb_args, capture_output=True)

        if p.returncode == 1:
            print("[INFO] Collision encountered...retrying")
            continue

        not_yet_generated = False
        if p.returncode != 0:
            print("[ERROR] rpmb.py failed")
            sys.exit(1)

    return (rp_path, rp_file_id)


def gen_save_path(name, seed, dataset):
    return dlrm_output_dir + "/{}/model/{}_{}.m".format(dataset, name, seed)


def gen_log_path(name, seed, dataset):
    return dlrm_output_dir + "/{}/log/{}_{}.log".format(dataset, name, seed)


class rp_config:
    def __init__(
        self, mlp_bot_partial, size_in, size_out, rp_file_tuple=None, rp_file_id=None
    ):
        self.mlp_bot_partial = mlp_bot_partial
        self.sparse_feat_size_in = size_in
        self.sparse_feat_size_out = size_out
        if not rp_file_tuple:
            # print("[INFO]", "Generating new RP matrix")
            rp_file_tuple = gen_rp_mat(size_in, size_out)
        self.rp_file = rp_file_tuple[0]
        self.rp_file_id = rp_file_tuple[1]

    def get_name(self):
        return "dlrm_rp_{}_{}_{}".format(
            self.rp_file_id, self.sparse_feat_size_in, self.sparse_feat_size_out
        )

    def extra_args(self):
        args = {}
        args["arch-mlp-bot"] = self.mlp_bot_partial + str(self.sparse_feat_size_out)
        args["arch-sparse-feature-size"] = self.sparse_feat_size_in
        args["enable-rp"] = None
        args["rp-file"] = self.rp_file
        return args


class LinConfig:
    def __init__(self, mlp_bot_partial, size_in, size_out, init_fn):
        self.mlp_bot_partial = mlp_bot_partial
        self.sparse_feat_size_in = size_in
        self.sparse_feat_size_out = size_out
        self.init_fn = init_fn

    def get_name(self):
        return "dlrm_lin_{}_{}_{}".format(self.sparse_feat_size_in,
                                        self.sparse_feat_size_out,
                                        self.init_fn)

    def extra_args(self):
        args = {}
        args["arch-mlp-bot"] = self.mlp_bot_partial + str(self.sparse_feat_size_out)
        args["arch-sparse-feature-size"] = self.sparse_feat_size_in
        args["linp-init"] = self.init_fn
        args["enable-linp"] = None
        return args

class VanillaConfig():
    def __init__(self, mlp_bot_partial, size):
        self.mlp_bot_partial = mlp_bot_partial
        self.size = size

    def get_name(self):
        return "dlrm_vanilla_{}".format(self.size)

    def extra_args(self):
        args = {}
        return args

class ProcLogWrapper:
    def __init__(self, proc, logfile, name):
        self.proc = proc
        self.log = logfile
        self.active = True
        self.ret = None
        self.name = name

    def poll(self):
        if self.active:
            self.ret = self.proc.poll()
            if self.ret is not None:
                self.active = False
                # cleanup file handle
                self.log.close()


def execute_queue(queue, dataset):

    max_active = 1
    proc_wrappers = []
    active_count = 0

    start = datetime.datetime.now()

    inactive_status_lines = []
    while len(queue) > 0:
        plw = launch_config(queue.pop(), random.randint(0, 10000), dataset)
        proc_wrappers.append(plw)

        active_count += 1

        while active_count == max_active:
            time.sleep(5)
            active_count = 0
            active_status_lines = []
            to_delete = []

            for plw in proc_wrappers:
                plw.poll()
                if plw.active:
                    active_status_lines.append(plw.name)
                    active_count += 1
                else:
                    inactive_status_lines.append(plw.name)
                    to_delete.append(plw)

            for plw in to_delete:
                proc_wrappers.remove(plw)

            lines_printed = 0
            print(
                "Status: Jobs in queue: {}, Total runtime: {}".format(
                    len(queue), datetime.datetime.now() - start
                )
            )
            print("{}/{} processes running".format(active_count, max_active))
            lines_printed += 2

            print("----Queued----")
            lines_printed += 1
            for cf in queue[-10:]:
                print("\t{}".format(cf.get_name()))
                lines_printed += 1
            if len(queue) > 10:
                print("\t...")
                lines_printed += 1

            print("----Active----")
            lines_printed += 1
            for sl in active_status_lines:
                print("\t{}".format(sl))
                lines_printed += 1

            # last 10
            print("---Complete---")
            lines_printed += 1
            for sl in inactive_status_lines[-10:]:
                print("\t{}".format(sl))
                lines_printed += 1

            if len(inactive_status_lines) > 10:
                print("\t...")
                lines_printed += 1

            print("\033c", end="")

        # print("\033[{}F".format(lines_printed), end="")


def launch_config(config, seed, dataset, test_run=False, dry_run=False):

    log_path = gen_log_path(config.get_name(), seed, dataset)
    save_args = {"save-model": gen_save_path(config.get_name(), seed, dataset)}
    seed_args = {"numpy-rand-seed": seed}
    extra_args = config.extra_args()

    full_args_items = (
        list(generic_args.items())
        + list(extra_args.items())
        + list(save_args.items())
        + list(seed_args.items())
    )

    if test_run:
        full_args_items = full_args_items + list(test_run_args.items())

    if dataset == "avazu":
        full_args_items = full_args_items + list(avazu_args.items())
    elif dataset == "criteo-kaggle":
        full_args_items = full_args_items + list(criteo_kaggle_args.items())
    else:
        print("[ERROR] {} unknown".format(dataset))
        sys.exit(1)

    log = open(log_path, "w")

    log.write("args:\n")

    args = []
    for arg, val in full_args_items:
        log.write("{} : {}\n".format(arg, val))
        args.append("--" + arg)
        if val is not None:
            args.append(str(val))

    # if dry run: only log the args
    if dry_run:
        print("DRY RUN ARGS")
        print(args)
        log.write("DRY RUN ARGS\n")
        log.write(str(args))
        log.close()

        return None

    else:
        proc = subprocess.Popen(
            dlrm_exe + args,
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        return ProcLogWrapper(proc, log, config.get_name())


if __name__ == "__main__":

    queue = []

    reps = 5
    for in_size in [4,8,16,32,64,128]:
        for i in range(reps):
            queue.append(VanillaConfig("1-256-64-", in_size))

    execute_queue(queue, "avazu")
