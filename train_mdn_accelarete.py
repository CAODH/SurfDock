import copy
import math
import os
from functools import partial

import wandb
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from datasets.pdbbind import construct_loader
from utils.parsing import parse_train_args
from utils.training_mdn import train_mdn_epoch, test_mdn_epoch
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage
import datetime

def train(args, model, optimizer, scheduler,  ema_weights,train_loader, val_loader, t_to_sigma, run_dir,accelerator):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    early_stop_patience = args.mdn_early_stop_patience
    patience_count = 0
    logger.info("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0: logger.info("Run name: {}".foramt(args.run_name))
        logs = {}
        #################trainging ########################
        train_losses = train_mdn_epoch(model, train_loader, optimizer, device,accelerator,ema_weights)
        # accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"epoch【{epoch}】@{nowtime} --> train_metric=")
            logger.info("Epoch {}: Training loss {:.4f}"
                .format(epoch, train_losses['loss'],flush=True))
        # accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        ema_weights.store(model.parameters())
        if args.use_ema: ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
        ############### trainging end#######################
        val_losses = test_mdn_epoch(model, val_loader, device, accelerator,args.test_sigma_intervals)
        #####################
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"epoch【{epoch}】@{nowtime} --> eval_metric=")
            logger.info("Epoch {}: Validation loss {:.4f} "
                .format(epoch, val_losses['loss']))

        if not args.use_ema: ema_weights.copy_to(model.parameters())
        accelerator.wait_for_everyone()
        # ema weight state dict
        unwrapped_model = accelerator.unwrap_model(model)
        ema_state_dict = copy.deepcopy(unwrapped_model.state_dict() if device.type == 'cuda' else unwrapped_model.state_dict())
        # last model weight state dict
        ema_weights.restore(model.parameters())
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # ema_state_dict = copy.deepcopy(unwrapped_model.state_dict() if device.type == 'cuda' else unwrapped_model.state_dict())
        state_dict = unwrapped_model.state_dict() if device.type == 'cuda' else unwrapped_model.state_dict()
        if accelerator.is_local_main_process:
            # accelerator.wait_for_everyone()
            if args.wandb:
                logs.update({'train_' + k: v for k, v in train_losses.items()})
                logs.update({'val_' + k: v for k, v in val_losses.items()})
                logs['current_lr'] = optimizer.param_groups[0]['lr']
                wandb.log(logs, step=epoch + 1)
            
            # if args.inference_earlystop_metric in logs.keys() and \
            #         (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
            #         args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            #     best_val_inference_value = logs[args.inference_earlystop_metric]
            #     best_val_inference_epoch = epoch
            #     torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            #     torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))
            patience_count += 1
            if val_losses['loss'] <= best_val_loss:
                patience_count =0
                best_val_loss = val_losses['loss']
                best_epoch = epoch
                torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))
            if patience_count == early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if scheduler:
            if args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses['loss'])
        if accelerator.is_local_main_process:
            # accelerator.wait_for_everyone()
            # unwrapped_optimizer = accelerator.unwrap_model(optimizer)
            torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))
    if accelerator.is_local_main_process:

        logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
        logger.info("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))
    if args.wandb:
            wandb.finish()
# from accelerate.utils import DummyOptim, DummyScheduler, set_seed
def main_function():
    import typing
    args = parse_train_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            elif isinstance(value, typing.Dict):
                arg_dict[key] = value['value']
        # logger.info(value['value'])
            else:
                arg_dict[key] = value
        # args.config = args.config.name 
    # logger.info(args)
    args.run_name =args.run_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if accelerator.is_local_main_process:
        # args.run_name =args.run_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if args.wandb:
            wandb.login(key = 'your key')
            
            wandb.init(
                entity='SurfDock',
                settings=wandb.Settings(start_method="fork"),
                project=args.project,
                name=args.run_name ,
                dir = args.wandb_dir,
                config=args
            )
            # wandb.log({'numel': numel})
    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)
    model = get_model(args, device, t_to_sigma=t_to_sigma,model_type = args.model_type)
    # get_model(confidence_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
    #                                 mdn_mode=True)
    optimizer, scheduler = get_optimizer_and_scheduler(args,model, accelerator,scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
    ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)
    #################################################
    if args.restart_dir:
        try:
            dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
            if args.restart_lr is not None: dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(dict['optimizer'])
            model.load_state_dict(dict['model'], strict=True)
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(dict['ema_weights'], device=device)
            logger.info(f"Restarting from epoch {dict['epoch']}")
        except Exception as e:
            logger.info(f"Exception: {e}")
            dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
            model.module.load_state_dict(dict, strict=True)
            logger.info("Due to exception had to take the best epoch and no optimiser")
    #################################################
    model = accelerator.prepare(model)
    optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    optimizer,train_loader, val_loader, scheduler)

    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model with {numel} parameters')

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device
    train(args, model, optimizer, scheduler, ema_weights,train_loader, val_loader, t_to_sigma, run_dir,accelerator)
    # if args.wandb:
    #     wandb.finish()
if __name__ == '__main__':
    from accelerate import Accelerator
    # from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(kwargs_handlers=[kwargs])
    from accelerate.utils import set_seed
    accelerator = Accelerator()
    device = accelerator.device
    set_seed(42)
    # accelerator = Accelerator(mixed_precision=mixed_precision)
    logger.info(f'device {str(accelerator.device)} is used!')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()
    # exit()
