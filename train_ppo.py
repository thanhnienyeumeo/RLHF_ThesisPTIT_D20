import click
import torch
from trainers import PPOTrainer
from configs import get_configs
from gpt import GPTActor, GPTRewardModel, GPTCritic
from dataset import StepDPOPromptsDataset, DahoasSFTStaticPromptsDataset, SafeRLHFPromptsDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train(batch_size, exp_name, actor_weights, critic_weights, debug):
    cfg_critic = get_configs("gpt2-medium")
    cfg_actor = get_configs("gpt2-large")
    cfg_actor.actor_weights = actor_weights
    # 67% gpt2-medium sft lora
    cfg_critic.critic_weights = critic_weights
    # 68% gpt2-xl
    # cfg.critic_weights = "./runs/rm_1678230899/rm_1678230899_final.pt"
    # 63%
    # cfg.critic_weights = "./runs/rm_gpt2medium-batch8-full-sft_202303141545/rm_gpt2medium-batch8-full-sft_202303141545_final.pt"
    cfg_critic.reward_model_weights = cfg_critic.critic_weights
    cfg_actor.sft_model_weights = cfg_actor.actor_weights
    cfg_actor.batch_size = batch_size
    cfg_actor.total_epochs = 3
    cfg_actor.exp_name = exp_name

    actor = GPTActor.from_checkpoint(cfg_actor, cfg_actor.actor_weights).cuda()
    sft_model = GPTActor.from_checkpoint(cfg_actor, cfg_actor.sft_model_weights).cuda()

    
    critic = GPTCritic.from_checkpoint(cfg_critic, cfg_critic.critic_weights).cuda()
    reward_model = GPTRewardModel.from_checkpoint(
        cfg_critic, cfg_critic.reward_model_weights).cuda()
    import pickle
    import numpy as np
    run_name = critic_weights[:critic_weights.rindex('\\')]
    epoch = 0
    neg_scores_cpu = pickle.load(open(f'{run_name}/neg_scores_{epoch}.pkl', 'rb'))
    pos_scores_cpu = pickle.load(open(f'{run_name}/pos_scores_{epoch}.pkl', 'rb'))
    mean = (np.sum(neg_scores_cpu) + np.sum(pos_scores_cpu))/(len(neg_scores_cpu) + len(pos_scores_cpu))
    std = np.sqrt((np.sum((neg_scores_cpu - mean)**2) + np.sum((pos_scores_cpu - mean)**2))/(len(neg_scores_cpu) + len(pos_scores_cpu)))
    # dataset = DahoasSFTStaticPromptsDataset(block_size=1024,
    #                                         max_examples=None,
    #                                         tokenizer_name="tiktoken/gpt2")
    # dataset = SafeRLHFPromptsDataset(block_size=1024,
    dataset = StepDPOPromptsDataset(block_size=1024,
                                            max_examples=None,
                                            tokenizer_name="tiktoken/gpt2")
    trainer = PPOTrainer(cfg_actor, actor, critic, reward_model, sft_model, dataset, debug, mean, std)
    trainer.fit()


@click.command()
@click.option('--strategy', '-s')
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
@click.option('--actor', '-a')
@click.option('--critic', '-c')
@click.option('--debug', '-d', default=False)
# @click.option('--dataset', '-d', default="sft")
def main(strategy, batch_size, exp_name, actor, critic, debug):
    train(batch_size, exp_name, actor, critic, debug)


if __name__ == "__main__":
    main()
