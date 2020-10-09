import numpy as np
import argparse
import torch
from torch import autograd
import sys
import time
import gym


import data
from Discriminator import AIRL_func
import logger
import utils

from Policies import Policy, Policy_MCP, Policy_MCP2
from sandbox.rocky.tf.envs.base import TfEnv
from inverse_rl.envs.env_utils import CustomGymEnv

#python Train.py --learn_temperature --env_name "CustomAnt-v0" --airl_reward --policy_name "SAC"
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_policy(policy_name, state_dim, action_dim, max_action, args):
    if policy_name == 'SAC':
        return Policy.SAC(state_dim, action_dim, max_action, args)

    elif policy_name == 'SAC_MCP':
        return Policy_MCP.SAC(state_dim, action_dim, max_action, args)

    # reduced number of premitives to 4 for this task
    elif policy_name == 'SAC_MCP2':
        return Policy_MCP2.SAC(state_dim, action_dim, max_action, args)

    # TODO: test other policies
    assert 'Unknown policy: %s' % policy_name


def evaluate_policy(env, generator, tracker, predict_reward, num_episodes=10):
    tracker.reset('eval_episode_reward')
    tracker.reset('eval_episode_timesteps')
    tracker.reset('eval_episode_predicted_reward')
    sum_reward = 0
    sum_p_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False

        timesteps = 0
        while not done:
            with torch.no_grad():
                with utils.eval_mode(generator):
                    action = generator.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
            lprob = generator.compute_pdf(torch.FloatTensor(state).reshape(1, -1).to(device),
                                                  torch.FloatTensor(action).reshape(1, -1).to(device))
            #state, next_state, action, lprobs
            p_reward = predict_reward(torch.FloatTensor(state).reshape(1, -1).to(device),
                                      torch.FloatTensor(next_state.reshape(1, -1)).to(device),
                                      torch.FloatTensor(action).reshape(1, -1).to(device),
                                      lprob.reshape(-1, 1))
            sum_reward += reward
            sum_p_reward += p_reward.detach().cpu().numpy()[0][0]
            timesteps += 1
            state = next_state

    tracker.update('eval_episode_reward', sum_reward/num_episodes)
    tracker.update('eval_episode_predicted_reward', sum_p_reward/num_episodes)
    tracker.update('eval_episode_timesteps', timesteps)

    return sum_reward/num_episodes

def create_predict_reward(discriminator, args):
    def compute(state, next_state, action, lprobs):
        with torch.no_grad():
            with utils.eval_mode(discriminator):

                if args.state_only == True:
                    reward = discriminator.reward_func(state)
                else:
                    reward = discriminator.reward_func(torch.cat([state, action], dim=1))
            return reward
    return compute

# TODO: need to think of a way around
def compute_gradient_penalty(discriminator, expert_state, expert_next_state, expert_action, expert_lprobs,
                            policy_state, policy_next_state, policy_action, policy_lprobs, stats=None):


    def get_mixed_data(expert_data,policy_data):
        alpha = torch.rand(expert_data.size(0), 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True
        return mixup_data

    mixup_state = get_mixed_data(expert_state,policy_state)
    mixup_next_state = get_mixed_data(expert_next_state, policy_next_state)
    mixup_action = get_mixed_data(expert_action, policy_action)
    mixup_lprobs = get_mixed_data(expert_lprobs, policy_lprobs)

    disc, _ = discriminator.run(mixup_state, mixup_next_state, mixup_action, mixup_lprobs, critarion='Expert')
    ones = torch.ones(disc.size()).to(disc.device)
    grad = autograd.grad(
        outputs=disc,
        inputs=mixup_state,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    # https://github.com/EmilienDupont/wgan-gp/blob/master/training.py#L100
    grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + 1e-12)
    grad_pen = 10 * ((grad_norm - 1) ** 2).sum()
    return grad_pen


# parser.add_argument('--max_timesteps', default=1e6, type=int)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='AIRL')
    parser.add_argument('--policy_name', default='SAC', help='TD3')
    parser.add_argument(
        "--env_name", default="CustomAnt-v0")  # DisabledAnt-v0, CustomAnt-v0
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_timesteps', default=1e4, type=int)
    parser.add_argument('--eval_freq', default=5e3, type=int)
    parser.add_argument('--max_timesteps', default=1e6, type=int)  # careful when you change it during debugging
    parser.add_argument('--expl_noise', default=0.1, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--entropy_lambda', default=0.1, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)
    parser.add_argument('--num_traj', type=int, default=4)
    parser.add_argument('--subsamp_freq', type=int, default=20)
    parser.add_argument('--log_format', default='text', type=str)
    parser.add_argument('--load_weights', default=False, type=bool)
    parser.add_argument('--state_only', default=True, type=bool, help='Reward function is discriminator can be computed either r(s) or r(s,a)')
    parser.add_argument("--initial_temperature", default=0.2, type=float)  # SAC temperature
    parser.add_argument("--learn_temperature", action="store_true")  # Whether or not learn the temperature
    parser.add_argument("--compute_value_func", type=bool, default=True)
    parser.add_argument("--max_episode_timesteps", type=int, default=1000, help='Max steps allowed per epoch')
    parser.add_argument("--reward_log", action="store_true")
    parser.add_argument("--state_and_nextstate", action="store_true")
    parser.add_argument("--description", type=str, default="None")
    parser.add_argument("--save_weight_freq", default=5e4, type=int)
    parser.add_argument("--airl_reward", action="store_true")
    parser.add_argument("--empowerment", action="store_true")
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--disc_lr", type=float, default=3e-4)
    parser.add_argument('--use_lr', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.env_name == "DisabledAnt-v0" or "CustomAnt-v0":
        env = TfEnv(CustomGymEnv(args.env_name, record_video=False, record_log=False))
    else:
        env = gym.make(args.env_name)
        env.seed(args.seed)

    # Set seeds
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(seed)
    print ("---------------------------------------")
    print ("Algo: {}".format(args.algo))
    print ("State Only: %s" % (args.state_only))
    print ("Consider State and Next State: {}".format(args.state_and_nextstate))
    print ("Consider value function: {}".format(args.compute_value_func))
    print ("learn temperature: {}".format(args.learn_temperature))
    print ("Empowerment: {}".format(args.empowerment))
    print ("AIRL reward: {}".format(args.airl_reward))
    print ("Seed : %s" % (seed))
    print ("Algorithm: {} |Policy: {} | Environtment: {}".format(args.algo, args.policy_name, args.env_name))
    print ("---------------------------------------")

    if args.state_and_nextstate == args.state_only:
        sys.exit("Both of them can't be true")


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create replay buffers
    replay_buffer = data.ReplayBufferIRL()
    expert_path = './Expert_Trajectory/{}/learn_temp_{}/best_exp/expert_traj.npy'.format(args.env_name, args.learn_temperature)
    expert_buffer = utils.load_expert_data(expert_path, state_dim, action_dim)

    ####################
    # Initialize policy
    ####################
    generator = create_policy(args.policy_name, state_dim, action_dim, max_action, args)
    discriminator = AIRL_func(device, args, state_dim, action_dim)
    predict_reward = create_predict_reward(discriminator, args)

    absorbing_state = np.random.randn(state_dim)  # type: Union[ndarray, float]

    # Load Pre_trained Weights
    if args.load_weights == True:
        utils.load_weights(generator, discriminator, args)

    # Do a initial run
    generator.train()
    discriminator.train()

    ###################
    # Initialize logger
    ###################
    tracker = logger.StatsTracker()
    train_logger = logger.TrainLogger(args, args.log_format, [
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
        'discriminator_loss'])
    eval_logger = logger.EvalLogger(args, args.log_format)


    eval_logger.save_details(" Algo: {} \n Policy: {} \n Environment: {} \n State_only: {} \n Consider value function:{} \n seed: {} \n"
                        " max_episode_timesteps: {}  \n  Description: {} \n save_weight_freq: {} \n AIRL reward: {} \n"
                             "Empowerment: {} \n"
                        .format(args.algo, args.policy_name, args.env_name, args.state_only, args.compute_value_func, args.seed,
                                args.max_episode_timesteps, args.description, args.save_weight_freq, args.airl_reward, args.empowerment))


    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = True

    # As long iteration < 1e6
    while total_timesteps < args.max_timesteps:


        # ======================================================
        if done or episode_timesteps >= args.max_episode_timesteps:
        # ======================================================

            if total_timesteps != 0:
                train_logger.dump(tracker)

                # ===============================================
                # A. Update discriminator weights for 1000 iteration:
                # ===============================================
                for _i in range(episode_timesteps):

                    ###########################################
                    # 1. Sample expert and learner trajectories
                    ###########################################
                    #state, next_state, action, lprob, reward, done
                    expert_state, expert_next_state, expert_action, _ = expert_buffer.sample(args.batch_size)

                    # state, next_state, action, lprob, reward, done
                    policy_state, policy_next_state, policy_action, _ ,  _, _  = replay_buffer.sample(args.batch_size)

                    #####################
                    # convert to Tensors
                    #####################
                    expert_state = torch.FloatTensor(expert_state).to(device)
                    expert_next_state = torch.FloatTensor(expert_next_state).to(device)
                    expert_action = torch.FloatTensor(expert_action).to(device)
                    expert_lprobs = generator.compute_pdf(expert_state, expert_action)


                    policy_state = torch.FloatTensor(policy_state).to(device)
                    policy_next_state = torch.FloatTensor(policy_next_state).to(device)
                    policy_action = torch.FloatTensor(policy_action).to(device)
                    policy_lprobs = generator.compute_pdf(policy_state, policy_action)


                    # ========================
                    # 2. Feed to Discriminator
                    # ========================
                    expert_D, expert_loss = discriminator.run(
                         expert_state, expert_next_state, expert_action, expert_lprobs, critarion='Expert')

                    policy_D, policy_loss = discriminator.run(
                         policy_state, policy_next_state, policy_action, policy_lprobs, critarion='Policy')




                    if total_timesteps % 5000 == 0 and _i == 1:
                        d_real_acc = torch.mean(torch.sigmoid(expert_D)).detach().cpu().numpy()
                        d_fake_acc = torch.mean(torch.sigmoid(policy_D)).detach().cpu().numpy()

                        print('---------------------------------------------------------------------')
                        print('Expert loss = {} | Learner loss = {}'.format(expert_loss, policy_loss))
                        print('Expert Prob = {} | Learner prob = {} (how confident agent is about the action being taken by expert)'.format(d_real_acc, d_fake_acc))


                    # ===================
                    # total computed loss
                    # ===================

                    discriminator_loss = expert_loss + policy_loss
                    tracker.update('discriminator_loss', discriminator_loss.item(), args.batch_size)
                    discriminator_loss /= args.batch_size

                    # ========================
                    # Compute gradient penalty
                    # =========================

                    grad_pen = compute_gradient_penalty(discriminator, expert_state, expert_next_state, expert_action, expert_lprobs,
                                                                policy_state, policy_next_state, policy_action, policy_lprobs)
                    grad_pen /= args.batch_size

                    if total_timesteps % 5000 == 0 and _i == 1:
                        print('Discriminator loss: {} | Gradient Penalty: {}'.format(discriminator_loss, grad_pen))

                        # update empowerment:
                    if args.empowerment:

                        discriminator.reward_optimizer.zero_grad()
                        discriminator_loss.backward()
                        grad_pen.backward()
                        discriminator.reward_optimizer.step()

                        discriminator.run(policy_state, policy_next_state, policy_action, policy_lprobs,
                                          critarion='empowerment_update', generator=generator)

                    else:
                    #if args.compute_value_func == True:
                        discriminator.value_optimizer.zero_grad()
                        discriminator.reward_optimizer.zero_grad()
                        discriminator_loss.backward()
                        grad_pen.backward()
                        discriminator.reward_optimizer.step()
                        discriminator.value_optimizer.step()





                # ===========================================
                # B. Update generator weights for 1000 iteration:
                # ===========================================
                """Working: input reward func ac discrimnator"""
                if args.policy_name == 'SAC' or 'SAC_MCP' or 'SAC_MCP2':
                    print('Training Generator -----')
                    generator.run(replay_buffer, episode_timesteps, tracker,
                                  args.batch_size, args.discount, args.tau, args.policy_freq,predict_reward,
                                  target_entropy=-action_dim if args.learn_temperature else None)
                    print('Done Training Generator -----')
                else: sys.exit("WARNING: Specify correct policy")


            # =========================================
            # Evaluate episode after every 5000 episode
            # ========================================
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                ep_r = evaluate_policy(env, generator, tracker, predict_reward)

                if total_timesteps%args.save_weight_freq == 0:
                    eval_logger.save_AIRL_weights(generator, discriminator,total_timesteps)
                    eval_logger.save_details('Avg episodic reward at {} timestep: {}'.format(ep_r,total_timesteps))

                eval_logger.dump(tracker)
                train_logger.dump(tracker)

                tracker.reset('train_episode_reward')
                tracker.reset('train_episode_timesteps')

            tracker.update('train_episode_reward', episode_reward)
            tracker.update('train_episode_timesteps', episode_timesteps)

            # Reset environment
            state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            tracker.update('num_episodes')
            tracker.reset('episode_timesteps')

        # =============================================================================================================
        #
        # if not done:
        #
        # =============================================================================================================

        ###############################################
        # 1. Take Action : Initially pick random action
        ###############################################



        with torch.no_grad():
            with utils.eval_mode(generator):
                _ , action, lprob = generator.sample_action(np.array(state))





        ########################
        # 2. Perform Action :
        ########################
        new_state, reward, done, _ = env.step(action)
        done_float = 0 if episode_timesteps + 1 == args.max_episode_timesteps else float(done)


        ########################################
        # 3. Store Observations in replay buffer :
        ########################################

        if done_float:
            # ( state, next_state, action, lprob, reward, done )
            replay_buffer.add((state, absorbing_state, action, lprob, reward, 0))
            replay_buffer.add((absorbing_state, absorbing_state, action, lprob, 0, 0))

        else:
            replay_buffer.add((state, new_state, action, lprob, reward, done_float))

        ###########################
        #  4. update Parameter :
        ###########################
        state = new_state
        episode_reward += reward
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        tracker.update('total_timesteps')
        tracker.update('episode_timesteps')

    # Done for 1e6 iterations

    ###################
    # Final evaluation
    ###################
    ep_r = evaluate_policy(env, generator, tracker, predict_reward)
    eval_logger.dump(tracker)  # Samin: Added scripts to save results in csv for every "eval.logger.dump()"
    train_logger.dump(tracker)
    ########################
    # Save the final weights
    ########################
    eval_logger.save_AIRL_weights(generator, discriminator, total_timesteps)
    eval_logger.save_details('Avg episodic reward at {} timestep: {}'.format(ep_r, total_timesteps))
    eval_logger.save_details("Total compute time: --- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()


