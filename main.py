import os
import uuid

import torch
from comet_ml import Experiment
from scripts.loggers import Logger
from scripts.utils import set_seed, make_config
from scripts.rlpyt_algo import CategoricalDQN_nce
from scripts.rlpyt_model import AtariCatDqnModel_nce
from scripts.rlpyt_override import _log_infos, log_diagnostics_custom, evaluate_agent, make_env
from rlpyt.agents.dqn.atari.atari_catdqn_agent import AtariCatDqnAgent
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger
from rlpyt.utils.logging.context import logger_context

from rlpyt.utils.launching.affinity import make_affinity, quick_affinity_code, affinity_from_code


"""
##############
#   ProcGen  #
##############

python main_procgen.py  --lambda_LL "0" --lambda_GL "0" --lambda_LG "0" --lambda_GG "1" --experiment-name "test" --env-name "procgen-bigfish-v0.500" \
                --n_step-return "7" --nce-batch-size "256" --horizon "10000" --algo "c51" --n-cpus "8" --n-gpus "1" --weight-save-interval "-1" --n_step-nce "-2" \
                --frame_stack "3" --nce_loss "InfoNCE_action_loss" --log-interval-steps=1000 --mode "serial"
"""

def build_and_train(args,game="", run_ID=0,config=None):
    """
    1. Parse the args object into dictionaries understood by rlpyt
    """
    config['env']['id'] = args.env_name
    config["eval_env"]["id"] = args.env_name

    config["eval_env"]["horizon"] = args.horizon
    config["env"]["horizon"] = args.horizon

    if 'procgen' in args.env_name:
        for k,v in vars(args).items():
            if args.env_name.split('-')[1] in k:
                config['env'][k] = v

    config['model']['frame_stack'] = args.frame_stack
    config['model']['nce_loss'] = args.nce_loss
    config['model']['algo'] = args.algo
    config['model']['env_name'] = args.env_name
    config['model']['dueling'] = args.dueling == 1
    config['algo']['double_dqn'] = args.double_dqn == 1
    config['algo']['prioritized_replay'] = args.prioritized_replay == 1
    config['algo']['n_step_return'] = args.n_step_return
    config['algo']['learning_rate'] = args.learning_rate

    config['runner']['log_interval_steps'] = args.log_interval_steps
    config['cmd_args'] = vars(args)
    
    """
    2. Create the CatDQN (C51) agent from custom implementation
    """

    agent = AtariCatDqnAgent(ModelCls=AtariCatDqnModel_nce,model_kwargs=config["model"], **config["agent"])
    algo = CategoricalDQN_nce(
            args=config['cmd_args'],
            ReplayBufferCls=None,
            optim_kwargs=config["optim"], **config["algo"]
            )

    if args.mode == 'parallel':
        affinity = make_affinity(
                    n_cpu_core=args.n_cpus,
                    n_gpu=args.n_gpus,
                    n_socket=1
                    # hyperthread_offset=0
                )

        """
        Some architecture require the following block to be uncommented. Try with and without.
        This is here to allow scheduling of non-sequential CPU IDs
        """
        # import psutil
        # psutil.Process().cpu_affinity([])
        # cpus = tuple(psutil.Process().cpu_affinity())
        # affinity['all_cpus'] = affinity['master_cpus'] = cpus
        # affinity['workers_cpus'] = tuple([tuple([x]) for x in cpus+cpus])
        # env_kwargs = config['env']

        sampler = GpuSampler(
                    EnvCls=make_env,
                    env_kwargs=config["env"],
                    CollectorCls=GpuWaitResetCollector,
                    TrajInfoCls=AtariTrajInfo,
                    eval_env_kwargs=config["eval_env"],
                    **config["sampler"]
                )
        """
        If you don't have a GPU, use the CpuSampler
        """
        # sampler = CpuSampler(
        #             EnvCls=AtariEnv if args.game is not None else make_env,
        #             env_kwargs=config["env"],
        #             CollectorCls=CpuWaitResetCollector,
        #             TrajInfoCls=AtariTrajInfo,
        #             eval_env_kwargs=config["eval_env"],
        #             **config["sampler"]
        #         )

    elif args.mode == 'serial':
        affinity = make_affinity(
                    n_cpu_core=1,  # Use 16 cores across all experiments.
                    n_gpu=args.n_gpus,  # Use 8 gpus across all experiments.
                    n_socket=1,
                    )

        """
        Some architecture require the following block to be uncommented. Try with and without.
        """
        # import psutil
        # psutil.Process().cpu_affinity([])
        # cpus = tuple(psutil.Process().cpu_affinity())
        # affinity['all_cpus'] = affinity['master_cpus'] = cpus
        # affinity['workers_cpus'] = tuple([tuple([x]) for x in cpus+cpus])
        # env_kwargs = config['env']

        sampler = SerialSampler(
                    EnvCls=make_env,
                    env_kwargs=config["env"],
                    # CollectorCls=SerialEvalCollector,
                    TrajInfoCls=AtariTrajInfo,
                    eval_env_kwargs=config["eval_env"],
                    **config["sampler"]
           )

    """
    3. Bookkeeping, setting up Comet.ml experiments, etc
    """
    folders_name = [args.output_dir,args.env_name,'run_'+args.run_ID]
    path = os.path.join(*folders_name)
    os.makedirs(path, exist_ok=True)

    experiment = Experiment(api_key='your_key',auto_output_logging=False, project_name='driml',workspace="your_workspace",disabled=True)
    experiment.add_tag('C51+DIM'if (args.lambda_LL > 0 or args.lambda_LG > 0 or args.lambda_GL > 0 or args.lambda_GG > 0) else  'C51')
    experiment.set_name( args.experiment_name )
    experiment.log_parameters(config)

    MinibatchRlEval.TF_logger = Logger(path, use_TFX=True, params=config, comet_experiment=experiment, disable_local=True)
    MinibatchRlEval.log_diagnostics = log_diagnostics_custom
    MinibatchRlEval._log_infos = _log_infos
    MinibatchRlEval.evaluate_agent = evaluate_agent

    """
    4. Define the runner as minibatch
    """
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )

    runner.algo.opt_info_fields = tuple(list(runner.algo.opt_info_fields) + ['lossNCE']+['action%d'%i for i in range(15)])
    name = args.mode+"_value_based_nce_" + args.env_name
    log_dir = os.path.join(args.output_dir, args.env_name)
    logger.set_snapshot_gap( args.weight_save_interval//config['runner']['log_interval_steps'] )

    """
    6. Run the experiment and optionally save network weights
    """

    with experiment.train():
        with logger_context(log_dir, run_ID, name, config,snapshot_mode=('last' if args.weight_save_interval == -1 else 'gap')): # set 'all' to save every it, 'gap' for every X it
            runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment-name', default='c51_procgen')
    parser.add_argument('--env-name', help='Procgen game', default='procgen-bigfish-v0.500')
    parser.add_argument('--n-cpus', help='number of cpus as workers', type=int, default=8)
    parser.add_argument('--n-gpus', help='number of gpus', type=int, default=1)
    parser.add_argument('--mode', help='mode (serial, async or parallel)', type=str, default='parallel',choices=['serial','async','parallel'])
    parser.add_argument('--output-dir', type=str, default='runs')
    parser.add_argument('--log-interval-steps', help='How often should the metrics be logged (in training steps)', type=int, default=100e3)
    parser.add_argument('--seed',default=0)
    ### C51 / Rainbow
    parser.add_argument('--algo', help='Algo type (only C51 for now)', default='c51',choices=['c51'])
    parser.add_argument('--dueling', type=int, default=0) # Never used
    parser.add_argument('--double-dqn', type=int, default=0) # Never used
    parser.add_argument('--n_step-return', type=int, default=7) # 7 seems to give stability
    parser.add_argument('--prioritized-replay', type=int, default=0) # Never used
    parser.add_argument('--learning-rate', type=float, default=2.5e-4) # Default from all value methods
    parser.add_argument('--horizon', type=int, default=27e3)
    parser.add_argument('--batch_size',type=int,default=256)
    ### NCE
    parser.add_argument('--nce-batch-size', type=int, default=256, help='NCE batch size, if different from RL batch size')
    parser.add_argument('--weight-save-interval', type=int,help='How often to save weights (default: every 500k steps). If set to -1, only best weight will be saved',default=500000)
    parser.add_argument('--frame_stack', type=int, default=4,help='Framestack (number of frames)')
    parser.add_argument('--run_ID', type=int, default=0,help='To start multiple runs with the same parameters on your cluster')
    parser.add_argument('--lambda_LL', type=float, default=0)
    parser.add_argument('--lambda_LG', type=float, default=0)
    parser.add_argument('--lambda_GL', type=float, default=0)
    parser.add_argument('--lambda_GG', type=float, default=0)
    parser.add_argument('--nce_loss', type=str, default='InfoNCE_action_loss')
    parser.add_argument('--score_fn', type=str,choices=['nce_scores_log_softmax','nce_scores_log_softmax_expanded'], default='nce_scores_log_softmax')
    parser.add_argument('--n_step-nce', type=int, default=1)

    args = parser.parse_args()

    args.run_ID = str(uuid.uuid1())
    set_seed(args.seed,torch.cuda.is_available())
    args.seed = str(args.seed)
    # multiprocessing.set_start_method('spawn')

    config = make_config(args.algo)
    build_and_train(
        args=args,
        game=None,
        run_ID=args.run_ID,
        config=config
    )
