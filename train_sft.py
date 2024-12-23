import click
import torch
from trainers import SFTTrainer
from gpt import GPT
from dataset import EYLSFTStaticDataset, MathDataset, MetaMathDataset
from configs import get_configs
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train(pretrain, batch_size, exp_name, epoch):
    device = 'cuda'
    cfg = get_configs("gpt2-large/dropout")
    cfg.max_steps = 100000 // batch_size
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    # assert pretrain == "huggingface"
    cfg.exp_name = exp_name
    optimizer = None
    step = 0
    if pretrain == "huggingface":
        model = GPT.from_pretrained(cfg)
    else:
        model = GPT.from_checkpoint(cfg, pretrain)
        ckp = torch.load(pretrain, map_location="cpu")
        optimizer = ckp["optimizer_state_dict"]
        step = ckp["step"]
    # train_ds = EYLSFTStaticDataset(block_size=1024,
    #                                split='train',
    #                                max_examples=None,
    #                                tokenizer_name="tiktoken/gpt2")
    # test_ds = EYLSFTStaticDataset(block_size=1024,
    #                               split='test',
    #                               max_examples=None,
    #                               tokenizer_name="tiktoken/gpt2")
    # train_ds = MathDataset(block_size=1024,
    #                                split='train',
    #                                max_examples=None,
    #                                tokenizer_name="tiktoken/gpt2")
    # test_ds = MathDataset(block_size=1024,
    #                               split='test',
    #                               max_examples=None,
    #                               tokenizer_name="tiktoken/gpt2")
    train_ds = MetaMathDataset(block_size=1024,
                                     split='train',
                                     max_examples=None,
                                    #  max_examples=10000,
                                     tokenizer_name="tiktoken/gpt2")
    leng = len(train_ds.tokens) // train_ds.block_size + 1
    cfg.max_steps = (leng * epoch) // batch_size
    # test_ds = MetaMathDataset(block_size=1024,
    #                                 split='test',
    #                                 max_examples=None,
    #                                 tokenizer_name="tiktoken/gpt2")
    test_ds = None
    trainer = SFTTrainer(cfg, device, model, train_ds, test_ds, epoch, optimizer, step)
    trainer.fit()


@click.command()
@click.option('--strategy', '-s')
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
@click.option('--epoch', '-e', default=2)
def main(strategy, pretrain, batch_size, exp_name, epoch):
    torch.manual_seed(1234)
    train(pretrain, batch_size, exp_name, epoch)


if __name__ == "__main__":
    main()
