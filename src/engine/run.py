import importlib
import os
from .args import parse_args, smart_instantiate
from ..utils import fix_seed

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

def locate(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f'Cannot import {module_name}: {e}')

def main():
    args = parse_args()
    fix_seed(args.seed)

    framework = args.framework.lower()
    mode = args.mode.lower()

    # map to module names in your repo
    if framework == 'dgl':
        trainer_module = 'src.engine.train_dgl'
        tester_module = 'src.engine.test_dgl'
    elif framework == 'pyg':
        trainer_module = 'src.engine.train_pyg'
        tester_module = 'src.engine.test_pyg'
    else:
        raise ValueError(f'Unknown framework: {framework}')

    if mode == 'train':
        mod = locate(trainer_module)
        # expect the trainer to expose a class named ATSPTrainerDGL or ATSPTrainerPyG named consistently
        TrainerClass = getattr(mod, 'ATSPTrainerDGL', None) or getattr(mod, 'ATSPTrainerPyG', None)
        if TrainerClass is None:
            raise AttributeError(f'{trainer_module} has no Trainer class (ATSPTrainerDGL/ATSPTrainerPyG)')
        trainer = TrainerClass(args, save_model=True)

        # load model factory
        models_mod = locate(f'src.models.models_{framework}')
        get_model = getattr(models_mod, f'get_{framework}_model', None)
        if get_model is None:
            raise AttributeError('Model factory not found in src.models (expected get_dgl_model/get_pyg_model/get_model)')
        model = get_model(args)  # pass args so factory can build with correct dims

        # run trials
        for trial in range(args.n_trials):
            print(f'=== Trial {trial} ===')
            results = trainer.train(model, trial_id=trial)
            # trainer.train saves checkpoints and results

    elif mode == 'test':
        # Load tester module
        mod = locate(tester_module)
        TesterClass = getattr(mod, 'ATSPTesterDGL', None) or getattr(mod, 'ATSPTesterPyG', None)
        if TesterClass is None:
            raise AttributeError(f'{tester_module} has no Tester class (ATSPTesterDGL/ATSPTesterPyG)')
        tester = TesterClass(args)

        # Load model factory
        models_mod = locate(f'src.models.models_{framework}')
        get_model = getattr(models_mod, f'get_{framework}_model', None)
        if get_model is None:
            raise AttributeError('Model factory not found in src.models (expected get_dgl_model/get_pyg_model)')
        model = get_model(args)
        results = tester.run_test(model)
        if args.results_dir:
            result_dir = args.results_dir
        else:
            base_dir = args.model_path
            if isinstance(base_dir, str) and base_dir.endswith('.pt'):
                base_dir = os.path.dirname(base_dir)
            result_dir = os.path.join(base_dir, f'test_atsp{args.atsp_size}')
        print(f"Testing completed. Results saved to {result_dir}/results_test_{args.atsp_size}.json")
    elif mode in ('arch', 'arch_search'):
        from src.arch_search.runner import ArchitectureSearchRunner

        searcher = ArchitectureSearchRunner(args)
        summary = searcher.run()
        best = summary.get("best_record")
        if best:
            print(f"Best architecture score={best['score']:.4f}, spec saved at {best['spec_path']}")
            if getattr(args, "combo_after_arch", False):
                from src.engine.search_all_combo import run_optuna_search

                args.model = "HGNASModel"
                args.architecture_path = os.path.abspath(best["spec_path"])
                if not getattr(args, "relation_subsets", None):
                    args.relation_subsets = [",".join(args.relation_types)]
                combo_trials = getattr(args, "combo_n_trials", None) or getattr(args, "n_trials", 1)
                print(
                    "Launching post-arch combo search: "
                    f"model={args.model}, trials={combo_trials}, architecture_path={args.architecture_path}, "
                    f"relation_subsets={args.relation_subsets}"
                )
                run_optuna_search(args, n_trials=combo_trials)

                # Optionally run batch_test after combo search
                if getattr(args, 'batch_test_after_arch', False):
                    from pathlib import Path
                    from src.engine.batch_test_search_models import main as batch_test_main

                    slurm_id = os.environ.get('SLURM_JOB_ID', 'no_slurm_job')
                    default_root = str(Path(__file__).resolve().parents[2] / 'search' / slurm_id)
                    search_root = getattr(args, 'batch_test_search_root', None) or default_root

                    # Build argv for batch_test_search_models.main()
                    argv = [
                        'batch_test_search_models',
                        '--search_root', str(search_root),
                        '--device', str(args.device),
                        '--time_limit', str(getattr(args, 'batch_test_time_limit', 5.0/30.0)),
                        '--perturbation_moves', str(getattr(args, 'batch_test_perturbation_moves', 30)),
                    ]
                    sizes = getattr(args, 'batch_test_sizes', None) or [100, 150, 250]
                    argv += ['--sizes'] + [str(s) for s in sizes]

                    # BooleanOptionalAction flags
                    if not getattr(args, 'batch_test_profile_flops', True):
                        argv.append('--no-profile-flops')
                    if getattr(args, 'batch_test_reuse_predictions', False):
                        argv.append('--reuse-predictions')
                    override_sizes = getattr(args, 'batch_test_override_sizes', None)
                    if override_sizes:
                        argv += ['--override_sizes'] + [str(s) for s in override_sizes]

                    print(f"\n[Auto] Launching batch_test: search_root={search_root}, sizes={sizes}\n")
                    import sys
                    _old_argv = sys.argv
                    try:
                        sys.argv = argv
                        batch_test_main()
                    finally:
                        sys.argv = _old_argv
        else:
            print("Architecture search finished but no successful evaluations were recorded.")
    else:
        raise ValueError(f'Unknown mode: {mode}')

if __name__ == '__main__':
    main()
