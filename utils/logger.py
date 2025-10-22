import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
import torch
import math
from tensorboardX import SummaryWriter

def log_to_screen(init_value, best_value, reward, costs_history, search_history,
                  batch_size, dataset_size, T):
    # reward
    print('\n', '-'*70)
    print('Avg total reward:'.center(35), '{:<10f} +- {:<10f}'.format(
        reward.sum(1).mean(), torch.std(reward.sum(1)) / math.sqrt(batch_size)))
    print('Avg step reward:'.center(35), '{:<10f} +- {:<10f}'.format(
        reward.mean(), torch.std(reward) / math.sqrt(batch_size)))

    # cost
    print('-'*70)
    print('Avg init cost:'.center(35), '{:<10f} +- {:<10f}'.format(
        init_value.mean(), torch.std(init_value) / math.sqrt(batch_size)))
    for per in range(20,100,20):
        cost_ = costs_history[:,round(T*per/100)]
        print(f'Avg cost after {per}% steps:'.center(35), '{:<10f} +- {:<10f}'.format(
            cost_.mean(),
            torch.std(cost_) / math.sqrt(batch_size)))
    # best cost
    print('-'*70)

    for per in range(20,100,20):
        cost_ = search_history[:,round(T*per/100)]
        print(f'Avg best cost after {per}% steps:'.center(35), '{:<10f} +- {:<10f}'.format(
            cost_.mean(),
            torch.std(cost_) / math.sqrt(batch_size)))
    print('Avg final best cost:'.center(35), '{:<10f} +- {:<10f}'.format(
        best_value.mean(), torch.std(best_value) / math.sqrt(batch_size)))

    print('-'*70, '\n')

def log_to_val(tb_logger, collect_gbest, R, epoch_id):
    # len1=cost_rollout.shape[0]
    
    # todo
    # for i in range(len1):
    #     tb_logger.add_scalars('validation/Optimization_curve_epoch_{}'.format(epoch_id),{'MTRL':cost_rollout[i].item()},i)
    
    tb_logger.add_scalars('performance_overtime',{'MTRL':collect_gbest},epoch_id)
    tb_logger.add_scalars('valication_Return',{'MTRL':R},epoch_id)

def log_to_tb_train(tb_logger, agent, Reward , R ,critic_out, ratios, bl_val_detached, total_cost, grad_norms, reward, entropy, approx_kl_divergence,
                    reinforce_loss, baseline_loss, log_likelihood,  show_figs, mini_step):

    tb_logger.add_scalar('learnrate_pg', agent.optimizer.param_groups[0]['lr'], mini_step)
    # avg_cost = (total_cost).mean().item()
    # tb_logger.add_scalar('train/avg_cost', avg_cost, mini_step)
    tb_logger.add_scalar('train/Target_Return', Reward.mean().item(), mini_step)
    tb_logger.add_scalar('train/Target_Return_changed', R.mean().item(), mini_step)
    tb_logger.add_scalar('train/Critic_output',critic_out.mean().item(),mini_step)
    tb_logger.add_scalar('train/ratios', ratios.mean().item(), mini_step)
    avg_reward = torch.stack(reward, 0).sum(0).mean().item()
    max_reward = torch.stack(reward, 0).max(0)[0].mean().item()
    tb_logger.add_scalar('train/avg_reward', avg_reward, mini_step)
    # tb_logger.add_scalar('train/init_cost', initial_cost.mean(), mini_step)
    tb_logger.add_scalar('train/max_reward', max_reward, mini_step)
    grad_norms, grad_norms_clipped = grad_norms
    tb_logger.add_scalar('loss/actor_loss', reinforce_loss.item(), mini_step)
    tb_logger.add_scalar('loss/nll', -log_likelihood.mean().item(), mini_step)
    tb_logger.add_scalar('train/entropy', entropy.mean().item(), mini_step)
    tb_logger.add_scalar('train/approx_kl_divergence', approx_kl_divergence.item(), mini_step)
    tb_logger.add_histogram('train/bl_val',bl_val_detached.cpu(),mini_step)

    tb_logger.add_scalar('grad/actor', grad_norms[0], mini_step)
    tb_logger.add_scalar('grad_clipped/actor', grad_norms_clipped[0], mini_step)
    tb_logger.add_scalar('loss/critic_loss', baseline_loss.item(), mini_step)

    tb_logger.add_scalar('loss/total_loss', (reinforce_loss+baseline_loss).item(), mini_step)

    tb_logger.add_scalar('grad/critic', grad_norms[1], mini_step)
    tb_logger.add_scalar('grad_clipped/critic', grad_norms_clipped[1], mini_step)

    if show_figs and mini_step % 1000 == 0:
        tb_logger.log_images('grad/actor',[plot_grad_flow(agent.actor)], mini_step)
        tb_logger.log_images('grad/critic',[plot_grad_flow(agent.critic)], mini_step)


def log_to_tb_train_wu(tb_logger, agent,  entropy, loss, mini_step):

    tb_logger.add_scalar('learnrate_pg', agent.optimizer.param_groups[0]['lr'], mini_step)
    tb_logger.add_scalar('loss/actor_loss', loss.item(), mini_step)
    tb_logger.add_scalar('train/entropy', entropy.mean().item(), mini_step)

