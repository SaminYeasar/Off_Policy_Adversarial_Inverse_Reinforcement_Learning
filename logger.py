from collections import OrderedDict
import json
import numpy as np
import pandas as pd
import csv
from utils import create_folder
import time
import torch

class IntegerMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0

    def update(self, n=1):
        self.n += n

    def compute(self):
        return self.n

    def __str__(self):
        return '%d' % self.compute()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    def compute(self):
        return self.sum / max(1, self.count)

    def __str__(self):
        return '%.03f' % self.compute()



class MovingAverageMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.vals = np.zeros(window_size)
        self.counts = np.zeros(window_size)
        self.pointer = 0

    def reset(self):
        self.vals.fill(0)
        self.counts.fill(0)
        self.pointer = 0

    def update(self, val, n=1):
        self.vals[self.pointer] = val
        self.counts[self.pointer] = n
        self.pointer = (self.pointer + 1) % self.window_size

    def compute(self):
        return self.vals.sum() / max(1., self.counts.sum())

    def __str__(self):
        return '%.03f' % self.compute()



class StatsTracker(object):
    def __init__(self, window_size=1000):
        self.meters = OrderedDict()
        self.meters['train_episode_reward'] = AverageMeter()
        self.meters['train_episode_timesteps'] = AverageMeter()
        self.meters['eval_episode_reward'] = AverageMeter()
        self.meters['eval_episode_predicted_reward'] = AverageMeter()
        self.meters['eval_episode_timesteps'] = AverageMeter()
        self.meters['eval_highest_reward'] = AverageMeter()       # this index is used to save the policy network that gives highest reward so far
        self.meters['train_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()

        self.meters['train_reward'] = MovingAverageMeter(window_size)
        self.meters['train_predicted_reward'] = MovingAverageMeter(window_size)
        self.meters['reward_pearsonr'] = MovingAverageMeter(window_size)
        self.meters['actor_loss'] = MovingAverageMeter(window_size)
        self.meters['critic_loss'] = MovingAverageMeter(window_size)
        self.meters['expert_policy_loss'] = MovingAverageMeter(window_size)
        self.meters['expert_policy_entropy'] = MovingAverageMeter(window_size)
        self.meters['gail_loss'] = MovingAverageMeter(window_size)
        self.meters['discriminator_loss'] = MovingAverageMeter(window_size)
        self.meters['expert_loss'] = MovingAverageMeter(window_size)
        self.meters['policy_loss'] = MovingAverageMeter(window_size)

        self.meters['total_timesteps'] = IntegerMeter()
        self.meters['num_episodes'] = IntegerMeter()
        self.meters['episode_timesteps'] = IntegerMeter()
        self.meters['epoch'] = IntegerMeter()

    def reset(self, name=None):
        if name is None:
            for name in self.meters:
                self.meters[name].reset()
        else:
            self.meters[name].reset()

    def update(self, name, *args):
        self.meters[name].update(*args)


class Logger(object):
    def __init__(self, args, format_type, keys=[]):
        self.format_type = format_type
        self.keys = keys
        self.log_type = None
        self.store = []
        self.args = args

        dir = './Results/{}/{}/{}/learn_temp_{}'.format(args.algo, args.policy_name, args.env_name, args.learn_temperature)
        # empowerment
        if args.empowerment: cat = 'Reward_Empowerment'
        else: cat = 'Reward_Mine'
        dir = '{}/{}'.format(dir, cat)

        if args.algo == 'AIRL_Retrain':
            dir = '{}_init{}_{}_loadgatingfunc_{}_learnactor_{}'.format(dir, args.initial_state, args.initial_runs, args.load_gating_func, args.learn_actor)

        self.dir = '{}/{}_{}'.format(dir, time.strftime('%y-%m-%d-%H-%M-%s'), args.seed)
        create_folder(self.dir)  # create_folder (directory)

    def save_details(self, text):
        with open("{}/Output.txt".format(self.dir), "a") as text_file:
            text_file.write(text)

    def _format_json(self, stats):
        return json.dumps(stats)

    def _format_text(self, stats):
        pieces = []
        for key in stats:
            pieces.append('%s: %s' % (key, stats[key]))
        return '| ' + (' | '.join(pieces))

    def _format(self, stats):
        if self.format_type == 'json':
            return self._format_json(stats)
        elif self.format_type == 'text':
            return self._format_text(stats)
        assert False, 'unknown log_type: %s' % self.format_type

    def dump(self, tracker, session='Train'):
        stats = OrderedDict()
        stats['type'] = self.log_type
        for key in self.keys:
            stats[key] = str(tracker.meters[key])
        print(self._format(stats))

        # save results when doing evaluation during training period
        if self.log_type == 'eval' and session == 'Train':
            self.store.append([x for x in stats.values()])
            self.save_results(self.log_type, session)
        if self.log_type == 'train' and session == 'Train':
            self.store.append([x for x in stats.values()])
            self.save_results(self.log_type, session)
        if self.log_type == 'eval' and session == 'Test':
            self.store.append([x for x in stats.values()])
            self.save_results(self.log_type, session)

    # Samin: Added this to save results
    def save_results(self, log_type, session):
        df = pd.DataFrame.from_records(self.store)
        columns = ["type"]
        for key in self.keys:
            columns.append(key)

        #columns = [key for key in self.keys]
        df.columns = columns

        # # create directory
        # dir = './Results/{}/{}/{}/{}'.format(args.algo, args.policy_name, args.env_name, time.strftime('%y-%m-%d-%H-%M-%s'))
        # create_folder(dir)   # create_folder (directory)
        # save csv
        df.to_csv('{}/During_{}_{}.csv'.format(self.dir, session, log_type))
        print('saved_{}_results at {} '.format(log_type,self.dir))


    def save_weights(self, policy, discriminator, timestep):
        torch.save(policy.actor.state_dict(), '{}/{}_actor.pth'.format(self.dir, timestep))
        if self.args.policy_name == "SAC_MCP" or self.args.policy_name == "SAC_MCP2":
            torch.save(policy.gating_func.state_dict(), '{}/{}_gating.pth'.format(self.dir,timestep))
        torch.save(policy.critic.state_dict(),'{}/{}_critic.pth'.format(self.dir, timestep))
        torch.save(discriminator.reward_func.state_dict(),'{}/{}_discriminator_reward.pth'.format(self.dir, timestep))

    def save_AIRL_weights(self, policy, discriminator, timestep):
        torch.save(policy.actor.state_dict(), '{}/{}_actor.pth'.format(self.dir, timestep))
        print(self.args.policy_name)
        if self.args.policy_name == "SAC_MCP" or self.args.policy_name =="SAC_MCP2":
            torch.save(policy.gating_func.state_dict(), '{}/{}_gating.pth'.format(self.dir, timestep))
        torch.save(policy.critic.state_dict(),'{}/{}_critic.pth'.format(self.dir, timestep))
        torch.save(discriminator.reward_func.state_dict(),'{}/{}_discriminator_reward.pth'.format(self.dir, timestep))
        if self.args.empowerment == False:
            torch.save(discriminator.value_func.state_dict(), '{}/{}_discriminator_value.pth'.format(self.dir, timestep))
        else:
            torch.save(discriminator.qvar.state_dict(),'{}/{}_discriminator_qvar.pth'.format(self.dir, timestep))
            torch.save(discriminator.Empowerment.state_dict(),'{}/{}_discriminator_Empowerment.pth'.format(self.dir, timestep))


class TrainLogger(Logger):
    def __init__(self, args, format_type, init_keys=None):
        keys = init_keys or [
            'total_timesteps',
            'num_episodes',
            'episode_timesteps',
            'train_episode_reward',
            'train_episode_timesteps',
            'train_reward',
            'train_predicted_reward',
            'reward_pearsonr',
            'actor_loss',
            'critic_loss',
            'expert_policy_loss',
            'expert_policy_entropy'
        ]
        super(TrainLogger, self).__init__(args, format_type, keys)
        self.log_type = 'train'


class EvalLogger(Logger):
    def __init__(self, args, format_type, init_keys=None):
        keys = init_keys or [
            'total_timesteps',
            'num_episodes',
            'episode_timesteps',
            'eval_episode_reward',
            'eval_episode_timesteps',
            'eval_episode_predicted_reward'
        ]
        super(EvalLogger, self).__init__(args,format_type, keys)
        self.log_type = 'eval'
