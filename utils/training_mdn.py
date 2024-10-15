# import copy
import numpy as np
from tqdm import tqdm
import torch
import gc
from torch.distributions import Normal
from loguru import logger
# from training import AverageMeter
# from mdn_utils import mdn_loss_fn
def mdn_loss_fn(pi, sigma, mu, y,dist_threhold=7.0,eps = 1e-10):
    mu = torch.clip(torch.nan_to_num(mu,0.0),min=1e-6)
    sigma = torch.clip(torch.nan_to_num(sigma,0.0),min=1e-6)
    pi = torch.clip(torch.nan_to_num(pi,0.0),min=1e-6)
    
    """calculate the mdn """
    normal = Normal(mu.real, sigma.real)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -torch.logsumexp(torch.log(pi.real + eps) + loglik, dim=1)
    loss = loss[torch.where(y <= dist_threhold)[0]]
    loss = loss.mean()
    return torch.nan_to_num(loss,0.0)
# def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
#     normal = Normal(mu, sigma)
#     #loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
#     #loss = th.sum(loss * pi, dim=1)
#     #loss = -th.log(loss)
#     loglik = normal.log_prob(y.expand_as(normal.loc))
#     loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)
#     return loss
import torch as th

def mdn_loss_fn_min_diatance_atom(pi, sigma, mu, y,dist_threhold=7.0,eps = 1e-10,topN = 1):
    mu = torch.clip(torch.nan_to_num(mu,0.0),min=1e-6)
    sigma = torch.clip(torch.nan_to_num(sigma,0.0),min=1e-6)
    pi = torch.clip(torch.nan_to_num(pi,0.0),min=1e-6)

    """use ca- pose to calculate the mdn """
    normal = Normal(mu.real, sigma.real)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -torch.logsumexp(torch.log(pi.real + eps) + loglik, dim=1)
    loss = loss[torch.where(y <= dist_threhold)[0]]
    loss = loss.mean()
    return loss
def calculate_probablity(pi, sigma, mu, y,dist_threhold=5.0,eps = 1e-10):
    mu = torch.clip(torch.nan_to_num(mu,0.0),min=1e-6)
    sigma = torch.clip(torch.nan_to_num(sigma,0.0),min=1e-6)
    pi = torch.clip(torch.nan_to_num(pi,0.0),min=1e-6)
    normal = Normal(mu.real, sigma.real)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi.real + eps )
    prob = logprob.exp().sum(1)
    prob[torch.where(y > dist_threhold)[0]] = 0.
    return prob

class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                # logger.info(interval_idx[type_idx])
                # logger.info(v)
                # logger.info(interval_idx[type_idx], torch.ones(len(v)))
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out
def train_mdn_epoch(model, loader, optimizer, device,accelerator,ema_weights):
    model.train()
    meter = AverageMeter(['loss','mdn_loss_interaction', 'mdn_loss_ligand', 'atom_types_loss', 'bond_types_loss', 'residue_types_loss'],
                         unpooled_metrics=True)

    for data in loader:
    # for data in tqdm(loader, total=len(loader),disable=not accelerator.is_local_main_process):
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            logger.info("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            # pi, sigma, mu, dist = model(data)
            with accelerator.autocast():
                mdn_loss_interaction , mdn_loss_ligand , atom_types_loss , bond_types_loss , residue_types_loss = model(data)#mdn_loss_fn(pi, sigma, mu, dist)
                # logger.info(loss)
                loss = mdn_loss_interaction + mdn_loss_ligand + 0.001*atom_types_loss + 0.001*bond_types_loss + 0.001*residue_types_loss
                accelerator.backward(loss)
                optimizer.step()
            # gather all loss for plot
            mdn_loss_interaction= accelerator.gather(mdn_loss_interaction)
            mdn_loss_ligand= accelerator.gather(mdn_loss_ligand)
            atom_types_loss= accelerator.gather(atom_types_loss)
            bond_types_loss= accelerator.gather(bond_types_loss)
            residue_types_loss= accelerator.gather(residue_types_loss)
            loss= accelerator.gather(loss)
            # logger.info('loss val: ',loss.mean().cpu().detach())
            metrics = [loss.mean().cpu().detach(),mdn_loss_interaction.mean().cpu().detach() , mdn_loss_ligand.mean().cpu().detach() , atom_types_loss.mean().cpu().detach() , bond_types_loss.mean().cpu().detach() , residue_types_loss.mean().cpu().detach()]
            meter.add(metrics)
            # logger.info('loss train: ',loss.mean().cpu().detach())
            ema_weights.update(model.parameters())
            # meter.add([loss.mean().cpu().detach()])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.info('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                optimizer.zero_grad()
                del data
                # loss = 0.0*sum([p.sum() for p in model.parameters() if p.requires_grad])
                # accelerator.backward(loss)
                # optimizer.step()
                gc.collect()
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                logger.info('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                optimizer.zero_grad()
                del data
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    return meter.summary()


def test_mdn_epoch(model, loader, device,accelerator, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss','mdn_loss_interaction', 'mdn_loss_ligand', 'atom_types_loss', 'bond_types_loss', 'residue_types_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss'],
            unpooled_metrics=True, intervals=10)

    for data in loader:
        try:
            with torch.no_grad():
                    # pi, sigma, mu, dist,_ = model(data)
                with accelerator.autocast():
                    mdn_loss_interaction , mdn_loss_ligand , atom_types_loss , bond_types_loss , residue_types_loss,_ = model(data)#mdn_loss_fn(pi, sigma, mu, dist)
                    loss = mdn_loss_interaction + mdn_loss_ligand + 0.001*atom_types_loss + 0.001*bond_types_loss + 0.001*residue_types_loss
            mdn_loss_interaction= accelerator.gather(mdn_loss_interaction)
            mdn_loss_ligand= accelerator.gather(mdn_loss_ligand)
            atom_types_loss= accelerator.gather(atom_types_loss)
            bond_types_loss= accelerator.gather(bond_types_loss)
            residue_types_loss= accelerator.gather(residue_types_loss)
            loss= accelerator.gather(loss)
            # logger.info('loss val: ',loss.mean().cpu().detach())
            metrics = [loss.mean().cpu().detach(),mdn_loss_interaction.mean().cpu().detach() , mdn_loss_ligand.mean().cpu().detach() , atom_types_loss.mean().cpu().detach() , bond_types_loss.mean().cpu().detach() , residue_types_loss.mean().cpu().detach()]
            meter.add(metrics)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.info('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                gc.collect()
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                logger.info('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    # if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out