import os
import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from trainers import RewardModelTrainer, FSDPRewardModelTrainer, AcceleratorRewardModelTrainer
from configs import get_configs
from gpt import GPTRewardModel
from dataset import StepDPODataset,DahoasRMStaticDataset, AnthropicHHRLHFDataset


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_fsdp(rank, world_size, pretrain):
    print(f"Start rank {rank} with world size {world_size}")
    setup(rank, world_size)
    device = "cuda"
    cfg = get_configs("gpt2-xl")
    cfg.activation_checkpointing = True
    cfg.batch_size = 1
    rm = GPTRewardModel.from_pretrained(cfg)
    train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=20,
                                     tokenizer_name="tiktoken/gpt2")
    test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=20,
                                    tokenizer_name="tiktoken/gpt2")
    trainer = FSDPRewardModelTrainer(cfg,
                                     device,
                                     rm,
                                     train_ds,
                                     test_ds,
                                     total_epochs=1,
                                     rank=rank,
                                     world_size=world_size,
                                     finetune_method=False)
    trainer.fit()
    dist.barrier()
    cleanup()


def train_accelerate():
    device = "cuda"
    cfg = get_configs("gpt2-xl")
    rm = GPTRewardModel.from_pretrained(cfg)
    cfg.batch_size = 1
    train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=20,
                                     tokenizer_name="tiktoken/gpt2")
    test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=20,
                                    tokenizer_name="tiktoken/gpt2")
    trainer = AcceleratorRewardModelTrainer(cfg,
                                            device,
                                            rm,
                                            train_ds,
                                            test_ds,
                                            total_epochs=1,
                                            finetune_method=False)
    trainer.fit()


def train(pretrain, batch_size, exp_name, epoch, dataset):
    device = 'cuda'
    cfg = get_configs("gpt2-medium/dropout")
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    cfg.total_epochs = epoch
    cfg.exp_name = exp_name

    if pretrain == "huggingface":
        rm = GPTRewardModel.from_pretrained(cfg)
    else:
        rm = GPTRewardModel.from_checkpoint(cfg, pretrain)
    if dataset == 'AnthropicHHRLHFDataset':
      train_ds = AnthropicHHRLHFDataset(block_size=1024,
                                      split='train',
                                      max_examples=None,
                                      tokenizer_name="tiktoken/gpt2")
      test_ds = AnthropicHHRLHFDataset(block_size=1024,
                                      split='test',
                                      max_examples=None,
                                      tokenizer_name="tiktoken/gpt2")
    
    elif dataset == "StepDPO":
      
      train_ds = StepDPODataset(block_size=1024,
                                     split='train',
                                     max_examples=None,
                                     tokenizer_name="tiktoken/gpt2")
      test_ds = StepDPODataset(block_size=1024,
                                     split='test',
                                     max_examples=None,
                                     tokenizer_name="tiktoken/gpt2")
    else: 
      train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=None,
                                     tokenizer_name="tiktoken/gpt2")
      test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=None,
                                    tokenizer_name="tiktoken/gpt2")
    trainer = RewardModelTrainer(cfg, device, rm, train_ds, test_ds)
    trainer.fit()


@click.command()
@click.option('--strategy', '-s', default="naive")
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="math")
@click.option('--epoch', '-e', default=5)
@click.option('--dataset', '-d', default='StepDPO')
def main(strategy, pretrain, batch_size, exp_name, epoch, dataset):
    torch.manual_seed(1234)

    if strategy == "fsdp":
        WORLD_SIZE = torch.cuda.device_count()
        mp.spawn(train_fsdp,
                 args=(WORLD_SIZE, pretrain),
                 nprocs=WORLD_SIZE,
                 join=True)
    elif strategy == "accelerate":
        train_accelerate()
    elif strategy == "naive":
        train(pretrain, batch_size, exp_name, epoch, dataset)


if __name__ == "__main__":
    main()
