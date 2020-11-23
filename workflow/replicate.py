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
    # "dump-emb-init": None, #dump the newly intialized table
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


class RPConfig:
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
    def __init__(self, mlp_bot_partial, size_in, size_out, init_fn, extra={},
            extra_prefix=""):
        self.mlp_bot_partial = mlp_bot_partial
        self.sparse_feat_size_in = size_in
        self.sparse_feat_size_out = size_out
        self.init_fn = init_fn
        self.extra_kwargs = extra
        self.extra_prefix = extra_prefix

    def get_name(self):
        return "dlrm_lin{}_{}_{}_{}".format(
            self.extra_prefix, self.sparse_feat_size_in, self.sparse_feat_size_out, self.init_fn
        )

    def extra_args(self):
        args = {}
        args["arch-mlp-bot"] = self.mlp_bot_partial + str(self.sparse_feat_size_out)
        args["arch-sparse-feature-size"] = self.sparse_feat_size_in
        args["linp-init"] = self.init_fn
        args["enable-linp"] = None
        for k, v in self.extra_kwargs.items():
            args[k] = v
        return args


class VanillaConfig:
    def __init__(self, mlp_bot_partial, size, extra={}, extra_prefix=""):
        self.mlp_bot_partial = mlp_bot_partial
        self.size = size
        self.extra = extra
        self.extra_prefix = extra_prefix

    def get_name(self):
        return "dlrm_vanilla{}_{}".format(self.extra_prefix, self.size)

    def extra_args(self):
        args = {}
        args["arch-mlp-bot"] = self.mlp_bot_partial + str(self.size)
        args["arch-sparse-feature-size"] = self.size
        for k,v in self.extra.items():
            args[k] = v
        return args


class ConcatOGConfig:
    def __init__(self, mlp_bot_partial, size):
        self.mlp_bot_partial = mlp_bot_partial
        self.size = size

    def get_name(self):
        return "dlrm_concat_og_{}".format(self.size)

    def extra_args(self):
        args = {}
        args["arch-mlp-bot"] = self.mlp_bot_partial + str(self.size)
        args["arch-sparse-feature-size"] = self.size
        args["concat-og-features"] = None
        return args


class LinConcatOGConfig:
    def __init__(self, mlp_bot_partial, size_in, size_out, init_fn, extra={}):
        self.mlp_bot_partial = mlp_bot_partial
        self.sparse_feat_size_in = size_in
        self.sparse_feat_size_out = size_out
        self.init_fn = init_fn
        self.extra_kwargs = extra

    def get_name(self):
        return "dlrm_lin_cc_og_{}_{}_{}".format(
            self.sparse_feat_size_in, self.sparse_feat_size_out, self.init_fn
        )

    def extra_args(self):
        args = {}
        args["arch-mlp-bot"] = self.mlp_bot_partial + str(self.sparse_feat_size_out)
        args["arch-sparse-feature-size"] = self.sparse_feat_size_in
        args["linp-init"] = self.init_fn
        args["enable-linp"] = None
        args["concat-og-features"] = None
        for k, v in kwargs.items():
            args[k] = v

        return args


class UpDownConfig:
    def __init__(
        self,
        mlp_bot_partial,
        size_in,
        size_mid,
        size_out,
        down_mode,
        up_mode,
        init_fn="normal",
    ):
        self.mlp_bot_partial = mlp_bot_partial

        if down_mode == "rp":
            down_rp_file_tuple = gen_rp_mat(size_in, size_mid)
            self.rp_file_down = down_rp_file_tuple[0]
            self.rp_file_down_id = down_rp_file_tuple[1]

        if up_mode == "rp":
            up_rp_file_tuple = gen_rp_mat(size_mid, size_out)
            self.rp_file_up = up_rp_file_tuple[0]
            self.rp_file_up_id = up_rp_file_tuple[1]

        self.init_fn = init_fn
        self.up_mode = up_mode
        self.down_mode = down_mode
        self.size_in = size_in
        self.size_mid = size_mid
        self.size_out = size_out

    def get_name(self):
        base_name = "dlrm_up_down_{}_{}_{}_{}_{}".format(
            self.up_mode, self.down_mode, self.size_in, self.size_mid, self.size_out
        )

        if self.down_mode == "linp" or self.up_mode == "linp":
            base_name += "_{}".format(self.init_fn)

        return base_name

    def extra_args(self):
        args = {}
        args["arch-mlp-bot"] = self.mlp_bot_partial + str(self.size_out)
        args["arch-sparse-feature-size"] = self.size_in

        if self.down_mode == "rp":
            raise NotImplementedError
        elif self.down_mode == "linp":
            args["enable-linp"] = None
            args["linp-init"] = self.init_fn
            args["proj-down-dim"] = self.size_mid
        else:
            raise NotImplementedError

        if self.up_mode == "rp":
            raise NotImplementedError
        elif self.up_mode == "linp":
            args["enable-linp-up"] = None
            args["linp-init"] = self.init_fn
        else:
            raise NotImplementedError


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
        seed = random.randint(0, 10000)
        plw = launch_config(queue.pop(), seed, dataset)
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
    err = open(log_path + ".err", "w")

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
        err.close()

        return None

    else:
        proc = subprocess.Popen(
            dlrm_exe + args, stdout=log, stderr=err, universal_newlines=True,
        )

        err.close()

        return ProcLogWrapper(proc, log, config.get_name())


if __name__ == "__main__":

    work_queue = []

    reps = 3
    #stdevs = [0.480383, 0.095016, 0.089492, 0.055964, 0.008288, 0.003648]
    for i in range(reps):
        for in_size in [32, 16, 8]:
            work_queue.append(
                VanillaConfig(
                    "1-256-64-",
                    in_size,
                    extra={"arch-interaction-op": "none"},
                    extra_prefix="_deep_only"
                )
            )
            for out_size in [32, 16, 8]:
                if out_size <= in_size:
                    work_queue.append(
                        LinConfig(
                            "1-256-64-",
                            in_size,
                            out_size,
                            init_fn="normal",
                            extra={"arch-interaction-op": "none"},
                            extra_prefix="_deep_only"
                        )
                    )

        # for out_size in [32,64,128]:
        #    for in_size in [128]:
        #        if out_size > in_size:
        #            continue
        #        work_queue.append(LinConfig("1-256-64-", in_size, out_size, init_fn="normal"))

        # for out_size in [4,8,16,32,64,128]:
        #    for in_size in [16]:
        #        if out_size >= in_size:
        #            continue
        #        work_queue.append(LinConfig("1-256-64-", in_size, out_size, init_fn="normal"))

    execute_queue(work_queue, "avazu")
