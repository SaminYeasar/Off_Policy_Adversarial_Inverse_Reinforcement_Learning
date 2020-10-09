import numpy as np
import argparse
import torch
from torch import autograd
import sys
import time
import glob
import os
import data


from Discriminator import AIRL_func
import logger
import utils
from Policies import PolicyRetrain, PolicyRetrain_MCP, PolicyRetrain_MCP2
from sandbox.rocky.tf.envs.base import TfEnv
from inverse_rl.envs.env_utils import CustomGymEnv

start_time = time.time()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_policy(policy_name, state_dim, action_dim, max_action, args):

    if policy_name == 'SAC':
        return PolicyRetrain.SAC(state_dim, action_dim, max_action, args)

    elif policy_name == 'SAC_MCP':
        return PolicyRetrain_MCP.SAC(state_dim, action_dim, max_action, args)

    elif policy_name == 'SAC_MCP2':
        return PolicyRetrain_MCP2.SAC(state_dim, action_dim, max_action, args)

    # TODO: test other policies
    assert 'Unknown policy: %s' % policy_name


def start_random_state(env, state, args, expert_policy=None):
    if args.env_name == 'DisabledAnt-v0': len = 100
    elif args.env_name == 'PointMazeRight-v0': len = 50

    if expert_policy != None:
        for _ in range(np.random.randint(1, len, 1)[0]):
            with torch.no_grad():
                with utils.eval_mode(expert_policy):
                    action = expert_policy.select_action(np.array(state))

            state, reward, done, _ = env.step(action)
            if args.display:
                env.render()
    else:
        for _ in range(np.random.randint(1, len, 1)[0]):
            state, reward, done, _ = env.step(env.action_space.sample())
            if args.display:
                env.render()
    return state


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
            p_reward = predict_reward(torch.FloatTensor(state).reshape(1, -1).to(device),
                                      torch.FloatTensor(action.reshape(1, -1)).to(device))
            sum_reward += reward
            sum_p_reward += p_reward.detach().cpu().numpy()[0][0]
            timesteps += 1
            state = next_state

    tracker.update('eval_episode_reward', sum_reward/num_episodes)
    tracker.update('eval_episode_predicted_reward', sum_p_reward/num_episodes)
    tracker.update('eval_episode_timesteps', timesteps)

    ########################################################
    # Save the policy with highest reward during evaluation:
    ########################################################
    # if tracker.meters['eval_highest_reward'].sum <= sum_reward:
    #     tracker.update('eval_highest_reward', sum_reward)
        #utils.save_AIRL_weights(generator, discriminator, args)
    return sum_reward/num_episodes



def create_predict_reward(discriminator, args,eval_logger):
    def compute(state, action):
        with torch.no_grad():
            with utils.eval_mode(discriminator):
                # r(s)
                if args.state_only == True:
                    reward = discriminator.reward_func(state)
                # r(s,a)
                else:
                    reward = discriminator.reward_func(torch.cat([state, action], dim=1))

            if args.reward_log:
                r = torch.sigmoid(reward)
                reward = (r + (1e-12)).log() - (1 - r + (1e-12)).log()
                if (reward == float('inf')).sum() > 0:
                    print("reward inf")
                    eval_logger.save_details("WARNING: reward inf")
                    sys.exit("WARNING: reward inf")

            return reward.detach()

    return compute


def compute_gradient_penalty(discriminator, expert_state, expert_next_state, expert_action, expert_lprobs,
                             policy_state, policy_next_state, policy_action, policy_lprobs, stats=None):
    def get_mixed_data(expert_data, policy_data):
        alpha = torch.rand(expert_data.size(0), 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True
        return mixup_data

    mixup_state = get_mixed_data(expert_state, policy_state)
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
    parser.add_argument('--algo', default='AIRL_Retrain')
    parser.add_argument('--policy_name', default='SAC_MCP', help='TD3')
    parser.add_argument("--env_name", default="DisabledAnt-v0")  # DisabledAnt-v0, PointMazeRight-v0, CustomAnt-v0
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
    parser.add_argument('--state_only', default=True, type=bool,
                        help='Reward function is discriminator can be computed either r(s) or r(s,a)')
    parser.add_argument("--initial_temperature", default=0.2, type=float)  # SAC temperature
    parser.add_argument("--learn_temperature", action="store_true")        # Whether or not learn the temperature
    parser.add_argument("--compute_value_func", type=bool, default=True)
    parser.add_argument("--load_gating_func", action="store_true",
                        help='whether or not to use previous gating function')
    parser.add_argument("--learn_actor", action='store_true',
                        help='for SAC definitely need to relearn SAC \
                             but for MCP changing gating function should be enough \
                             both case critic needs to be learnt again')
    parser.add_argument("--prior_weight_loc", type=str, default='19-09-05-15-45-1567712719_0', help='Use the file name')
    parser.add_argument("--initial_runs", type=str, default='policy_sample')
                                                            # env_sample: env.action_space.sample()
                                                            # policy_sample
                                                            # expert prior
    parser.add_argument("--max_episode_timesteps", type=int, default=500, help='Max steps allowed per epoch')
    parser.add_argument("--initial_state", type=str, default='random', help='Where does the agent start from') #random
    parser.add_argument("--reward_log", action="store_true")
    parser.add_argument("--empowerment", action="store_true")
    parser.add_argument("--save_weight_freq", default=5e5, type=int)
    parser.add_argument("--display", action='store_true')

    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--disc_lr", type=float, default=3e-4)
    parser.add_argument('--use_lr', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # =========================
    # Initialize environment:
    # =========================
    env = TfEnv(CustomGymEnv(args.env_name, record_video=False, record_log=False))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # =============
    # Set seeds :
    # =============
    seed = args.seed
    # env.seed(seed)       # env seed doesn't work
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(seed)

    # ===============
    # print details:
    print ("---------------------------------------")
    print ("Algo: {}".format(args.algo))
    print ("State Only: %s" % (args.state_only))
    print ("Consider value function: {}".format(args.compute_value_func))
    print ("Seed : %s" % (seed))
    print ("Algorithm: {} |Policy: {} | Environtment: {}".format(args.algo, args.policy_name, args.env_name))
    print ("---------------------------------------")


    # =============================================
    # Initialize generator(policy) and disciminator:p
    # =============================================
    #                             (policy_name, state_dim, action_dim, max_action, args)
    generator = create_policy(args.policy_name, state_dim, action_dim, max_action, args)
    discriminator = AIRL_func(device, args, state_dim, action_dim)

    # ==========================
    # Load Pre_trained Weights:
    # ==========================
    if args.env_name == "DisabledAnt-v0":
        prior_env = "CustomAnt-v0"
        load_weight_at = "1000000"

    elif args.env_name == "PointMazeRight-v0":
        prior_env = "PointMazeLeft-v0"
        #load_weight_at = "500000"
        load_weight_at = "1000000"

    loc_ = glob.glob('Results/AIRL/{}/{}/learn_temp_{}/Reward_Mine/*_{}'.format(args.policy_name, prior_env, args.learn_temperature, args.seed))[0]

    if os.path.exists(loc_): print('Loaded file: {}'.format(loc_))
    else: print('ERROR: INCORRECT LOCATION OF PRIOR LEARNT WEIGHTS')

    generator.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(loc_,load_weight_at)))
    generator.critic.load_state_dict(torch.load('{}/{}_critic.pth'.format(loc_,load_weight_at)))
    if args.load_gating_func == True:
        # we would want to learn a new gating function
        generator.gating_func.load_state_dict(torch.load('{}/{}_gating.pth'.format(loc_,load_weight_at)))
    discriminator.reward_func.load_state_dict(torch.load('{}/{}_discriminator_reward.pth'.format(loc_,load_weight_at)))


    if args.initial_state == "random":
        # expert policy here is "SAC"
        expert_policy = create_policy('SAC', state_dim, action_dim, max_action, args)
        expert_policy.actor.load_state_dict(
             torch.load(glob.glob('Expert_Weights/SAC/{}/temp_{}/*_{}/actor.pth'.format(args.env_name, args.learn_temperature, args.seed))[0]))


    # ===================
    # Initialize logger:
    # ===================
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
    eval_logger.save_details('{}'.format(args))
    eval_logger.save_details("\n \n Algo: {} \n Policy: {} \n Environment: {} \n State_only: {} \n Consider value function:{} \n seed: {} \n"
                        " max_episode_timesteps: {} \n initial_state: {} \n initial_runs: {} \n"
                             "load_gating_func: {} \n reward_log: {} \n learn_actor: {} \n"
                        .format(args.algo, args.policy_name, args.env_name, args.state_only, args.compute_value_func, args.seed,
                                args.max_episode_timesteps, args.initial_state, args.initial_runs, args.load_gating_func, args.reward_log,
                                args.learn_actor))



    # ===================
    # other essentials:
    # ===================
    predict_reward = create_predict_reward(discriminator, args, eval_logger)  #for AIRL it's computed using neural net  "reward_func"
    absorbing_state = np.random.randn(state_dim)                 # type: Union[ndarray, float]
    replay_buffer = data.ReplayBufferIRL()                       # initialize replay buffers


    # evaluate AIRL
    ep_r = evaluate_policy(TfEnv(CustomGymEnv(prior_env, record_video=False, record_log=False)), generator, tracker, predict_reward)
    print('Performance of trained agent on {} : {} \n'.format(prior_env, ep_r))
    eval_logger.save_details('\n Performance of trained agent on {} : {} \n'.format(prior_env, ep_r))

    # =======================
    # Initialize parameters:
    # =======================
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = True

    # As long iteration < 1e6
    while total_timesteps < args.max_timesteps:

        # ================================================================

        if done or episode_timesteps >= args.max_episode_timesteps:

        # =================================================================

            if total_timesteps != 0:
                train_logger.dump(tracker)

                # ============================================================
                # (1) Update discriminator : equal no times of env interaction (episode_timesteps)
                # ============================================================

                # Not needed during retraining for transfer learning
                # as we're supposed to used the learnt weight here

                # ===============================================================
                # (2) Update generator : equal no times of env interaction (episode_timesteps)
                # ===============================================================

                print('Training Generator -----')
                generator.run(replay_buffer, episode_timesteps, tracker,
                              args.batch_size, args.discount, args.tau, args.policy_freq,
                              discriminator, predict_reward,
                              target_entropy=-action_dim if args.learn_temperature else None)

            # ==========================================
            # Evaluate episode after every 5000 episode:
            # ==========================================
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                ep_r = evaluate_policy(env, generator, tracker, predict_reward)


                if total_timesteps%args.save_weight_freq == 0:
                    eval_logger.save_AIRL_weights(generator, discriminator,total_timesteps)
                    eval_logger.save_details('Avg episodic reward at {} timestep: {}'.format(ep_r, total_timesteps))


                eval_logger.dump(tracker)
                train_logger.dump(tracker)
                replay_buffer.save_traj(filename='trajectory', dirr=eval_logger.dir)

                tracker.reset('train_episode_reward')
                tracker.reset('train_episode_timesteps')

            tracker.update('train_episode_reward', episode_reward)
            tracker.update('train_episode_timesteps', episode_timesteps)

            # Reset environment
            state = env.reset()
            if args.display:
                env.render()
            #if total_timesteps < args.start_timesteps:
            if args.initial_state == "random":
                state = start_random_state(env, state, args, expert_policy)
                # URP = uniform random policy
            elif args.initial_state == "random_URP":
                state = start_random_state(env, state, args)

            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            tracker.update('num_episodes')
            tracker.reset('episode_timesteps')


        # ========================================================
        #
        # if not done:
        #
        # =========================================================

        ###############################################
        # 1. Take Action : Initially pick random action
        ###############################################

        if total_timesteps < args.start_timesteps:

            if args.initial_runs == 'env_sample':
                action = env.action_space.sample()
                lprob = generator.compute_pdf(torch.FloatTensor(np.array(state)).unsqueeze(0).to(device),
                                              torch.FloatTensor(np.array(action)).unsqueeze(0).to(device)).data.cpu().numpy().flatten()

            elif args.initial_runs == 'policy_sample':
                with torch.no_grad():
                    #     #_, action, lprob = expert_policy.sample_action(np.array(state))
                    with utils.eval_mode(generator):
                        _, action, lprob = generator.sample_action(np.array(state))
                        # action = (action + np.random.normal(
                        #     0, args.expl_noise, size=env.action_space.shape[0])).clip(
                        #     env.action_space.low, env.action_space.high)
            # TODO: give a look what airl does
            # elif args.initial_runs == 'expert_prior':
            #     with torch.no_grad():
            #         _, action, lprob = expert_policy.sample_action(np.array(state))


        ###################################
        # 1. Take Action : using TD3 policy
        ###################################
        else:
            # no-noisy case
            with torch.no_grad():
                with utils.eval_mode(generator):
                    # mu, pi, log pi
                    """if using SAC_IRL then should use "mu" as it computes log prob of mu"""
                    # action = generator.select_action(np.array(state))
                    _, action, lprob = generator.sample_action(np.array(state))
            # # noisy case
            # if args.expl_noise != 0:
            #     action = (action + np.random.normal(
            #         0, args.expl_noise, size=env.action_space.shape[0])).clip(
            #         env.action_space.low, env.action_space.high)

        ########################
        # 2. Perform Action :
        ########################
        new_state, reward, done, _ = env.step(action)
        if args.display:
            env.render()
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
    replay_buffer.save_traj(filename='trajectory', dirr=eval_logger.dir)
    ########################
    # Save the final weights
    ########################
    eval_logger.save_AIRL_weights(generator, discriminator, total_timesteps)
    eval_logger.save_details('Avg episodic reward at {} timestep: {}'.format(ep_r, total_timesteps))
    eval_logger.save_details("Total compute time: --- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()


