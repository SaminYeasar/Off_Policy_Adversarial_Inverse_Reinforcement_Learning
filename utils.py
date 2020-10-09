import torch
import os
import numpy as np
class eval_mode(object):
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.prev = self.model.training
        self.model.train(False)

    def __exit__(self, *args):
        self.model.train(self.prev)
        return False


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def save_weights(policy, discriminator, args, directory='./preTrained'):
    # see if the folder exit if note create one
    create_folder(directory)
    print("Saving weights")
    torch.save(policy.actor.state_dict(), '{}/{}_{}_{}_actor.pth'.format(directory, args.algo, args.policy_name, args.env_name))
    torch.save(policy.critic.state_dict(), '{}/{}_{}_{}_critic.pth'.format(directory, args.algo, args.policy_name, args.env_name))
    torch.save(discriminator.state_dict(), '{}/{}_{}_{}_discriminator.pth'.format(directory, args.algo, args.policy_name, args.env_name))


def load_weights(policy, discriminator, args, directory='./preTrained'):
    if os.path.exists(directory):
        print("Loading PreTrained Weights")
        policy.actor.load_state_dict(torch.load('{}/{}_{}_{}_actor.pth'.format(directory, args.algo, args.policy_name, args.env_name)))
        policy.critic.load_state_dict(torch.load('{}/{}_{}_{}_critic.pth'.format(directory, args.algo, args.policy_name, args.env_name)))
        discriminator.load_state_dict(torch.load('{}/{}_{}_{}_discriminator.pth'.format(directory, args.algo, args.policy_name, args.env_name)))
    else:
        print("PreTrained Weights don't exists. Training Agent from scratch")


class load_expert_data(object):
    def __init__(self, dirr, state_dim, action_dim):
        super(load_expert_data, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dirr =dirr
        if not os.path.exists(self.dirr):
            print('Expert trajectory doesnt exist at: {}'.format(self.dirr))
        if os.path.exists(self.dirr):
            self.expert_traj = np.load(self.dirr)
            print('Loaded Expert Data')

    def sample(self, batch_size):
        sampled_data = self.expert_traj[np.random.randint(0, self.expert_traj.shape[0], batch_size), :]
        #expert_state_action = torch.FloatTensor(expert_state_action).to(self.device)
        l1 = self.state_dim
        l2 = self.state_dim
        l3 = self.action_dim
        l4 = 1
        l5 = 1
        expert_state = sampled_data[:, 0:l1]
        expert_next_state = sampled_data[:, l1:l1+l2]
        expert_action = sampled_data[:, l1+l2:l1+l2+l3]
        expert_lprob = sampled_data[:, l1+l2+l3:l1+l2+l3+l4]

        return expert_state, expert_next_state, expert_action, expert_lprob

def save_AIRL_weights(policy, discriminator, args, directory='./preTrained'):
    # see if the folder exit if note create one
    create_folder(directory)
    print("Saving weights")
    torch.save(policy.actor.state_dict(), '{}/{}_{}_{}_actor.pth'.format(directory, args.algo, args.policy_name, args.env_name))
    torch.save(policy.critic.state_dict(), '{}/{}_{}_{}_critic.pth'.format(directory, args.algo, args.policy_name, args.env_name))
    if args.policy_name == "SAC_MCP":
        torch.save(policy.gating_func.state_dict(),
                   '{}/{}_{}_{}_gating.pth'.format(directory, args.algo, args.policy_name, args.env_name))
    torch.save(discriminator.reward_func.state_dict(), '{}/{}_{}_{}_discriminator_reward.pth'.format(directory, args.algo, args.policy_name, args.env_name))
    torch.save(discriminator.value_func.state_dict(),
               '{}/{}_{}_{}_discriminator_value.pth'.format(directory, args.algo, args.policy_name, args.env_name))





def save_AIRL_weights2(policy, discriminator, args, directory='./preTrained_AIRL'):
    # see if the folder exit if note create one
    #create_folder(directory)
    print("Saving weights")
    torch.save(policy.actor.state_dict(), '{}/{}_{}_{}_actor.pth'.format(directory, args.algo, args.policy_name, args.env_name))
    torch.save(policy.critic.state_dict(), '{}/{}_{}_{}_critic.pth'.format(directory, args.algo, args.policy_name, args.env_name))
    torch.save(discriminator.reward_func.state_dict(), '{}/{}_{}_{}_discriminator_reward.pth'.format(directory, args.algo, args.policy_name, args.env_name))



def gauss_log_pdf(mean, log_diag_std, x):
    N, d = mean.shape
    cov =  np.square(np.exp(log_diag_std))
    diff = x-mean
    exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=1)
    norm_term = -0.5*d*np.log(2*np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=1)
    log_probs = norm_term + var_term + exp_term
    return log_probs #sp.stats.multivariate_normal.logpdf(x, mean=mean, cov=cov)