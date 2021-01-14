import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from QNetwork import QNetwork

from Autoencoder import Encoder, Decoder
# from testExplainability import verifyCodes, showDistancesFromGoal
from Node import Node
from system_conf import ACTION_SIZE, CODE_SIZE, Q_HIDDEN_NODES, BATCH_SIZE, REW_THRE, WINDOW, MODELS_DIR
from transition_model import Transition, TransitionDelta
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def normalize_state_Lunar_Lander(vec):
    min_vec = [-0.9948723, -0.25610653, -4.6786265, -1.7428349, -1.9317414, -2.0567605, 0.0, 0.0]
    max_vec = [0.99916536, 1.5253401, 3.9662757, 0.50701684, 2.3456542, 2.0852647, 1.0, 1.0]
    for i in range(len(vec)):
        vec[i] = (vec[i] - min_vec[i]) / (max_vec[i] - min_vec[i])
        vec[i] = min(vec[i], 1)
        vec[i] = max(vec[i], 0)
    return vec


class plan_node():
    def __init__(self, code_state, value):
        self.code_state = code_state
        self.value = value
        self.action_vec = []

    def add_action(self, a):
        self.action_vec.extend(a)


class Plan_RL_agent:

    def __init__(self, env, buffer, load_models = False, epsilon=0.75, Q_hidden_nodes = Q_HIDDEN_NODES, batch_size= BATCH_SIZE, rew_thre = REW_THRE, window = WINDOW, path_to_the_models = MODELS_DIR):

        self.path_to_the_models = path_to_the_models
        self.env = env
        self.action_size = ACTION_SIZE

        if load_models:
            self.load_models()
        else:
            self.encoder = Encoder(CODE_SIZE)
            self.decoder = Decoder(CODE_SIZE)
            self.trans_delta = TransitionDelta(CODE_SIZE, self.action_size)
            self.transition = Transition(self.encoder, self.decoder, self.trans_delta)
            self.network = QNetwork(env=env, encoder=self.encoder, n_hidden_nodes=Q_hidden_nodes)
        self.target_network = deepcopy(self.network)
        #self.f = open("res/planner_enc_DDQN.txt", "a+")
        self.buffer = buffer
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.window = window
        self.reward_threshold = rew_thre
        self.initialize()
        self.action = 0
        self.step_count = 0
        self.cum_rew = 0
        self.timestamp = 0
        self.episode = 0
        self.difference = 0
        self.different_codes = 0
        self.A = [to_categorical(i, self.action_size) for i in range(self.action_size)]
        #self.transition_losses = []

    def expandFunc(self, x, a):
        _, s_prime = self.trans_delta(x, torch.from_numpy(a).type(torch.FloatTensor).to(device))
        return s_prime

    def vFunc(self, x):
        v0 = self.network.get_enc_value(x)
        return torch.max(v0).to('cpu').detach().numpy()

    def findPlan(self, node):
        # caso base
        if node.sons == []:
            return [node.a], node.v

        smax = -10000
        bestp = []
        for n in node.sons:
            p, s = self.findPlan(n)
            # print("plan p", p)
            # print("plan p", s)
            if s > smax:
                smax = s
                bestp = p

        return [node.a] + bestp, node.v + smax

    def limited_expansion(self, node, depth):  # , A): # , expandFunc, vFunc):
        if depth == 0:
            return

        for a in self.A:
            s_prime = self.expandFunc(node.x, a)
            node.expand(s_prime, self.vFunc(s_prime), a)

        for i in range(len(node.sons)):
            self.limited_expansion(node.sons[i], depth - 1)  # , A) # expandFunc, vFunc)

    def planner_action(self, depth=2):
        if np.random.random() < 0.05:
            return np.random.choice(self.action_size)
        origin_code = self.encoder(torch.from_numpy(self.s_0).type(torch.FloatTensor))
        origin_value = self.vFunc(origin_code)
        root = Node(origin_code, origin_value, to_categorical(0, self.action_size))

        self.limited_expansion(root, depth)  # , self.A)#, self.expFunc, self.vFunc)

        plan, sum_value = self.findPlan(root)

        return np.where(plan[1] == 1)[0][0]

    def is_diff(self, s1, s0):
        for i in range(len(s0)):
            if (s0[i] != s1[i]):
                return True
        return False

    def take_step(self, mode='train'):

        #actions = ['N', 'E', 'S', 'W']                              # <-----decomment for maze
        #s_1, r, done, _ = self.env.step(actions[self.action])       #
        s_1, r, done, _ = self.env.step(self.action)

        # enc_s1 = self.encoder(torch.from_numpy(np.asarray(s_1)).type(torch.FloatTensor))
        # enc_s0 = self.encoder(torch.from_numpy(np.asarray(self.s_0)).type(torch.FloatTensor).to('cuda'))
        # print("Reward = ", r)
        # if(self.is_diff(enc_s0,enc_s1)):
        # print("step passati = ", self.step_count - self.timestamp)
        # self.timestamp = self.step_count

        self.buffer.append(self.s_0, self.action, r, done, s_1)
        # self.cum_rew = 0

        if mode == 'explore':
            self.action = self.env.action_space.sample()
        else:
            # self.action = self.network.get_action(self.s_0)
            self.action = self.planner_action(depth=1)
        '''  # ADAPTIVE HORIZON
          if len(self.mean_training_rewards) < 1:
            self.action = self.env.action_space.sample()
          else:
                if self.mean_training_rewards[-1] < -5:
                  self.action = self.env.action_space.sample()
                else:
                  if self.mean_training_rewards[-1] < -3:
                    self.action = self.network.get_action(self.s_0)
                  else:
                    if self.mean_training_rewards[-1] < -2:
                      self.action = self.planner_action(depth= 1) #comment for Q-learning
                    else:
                      if self.mean_training_rewards[-1] < -1:
                        self.action = self.planner_action(depth= 1)
                      else:
                        self.planner_action(depth= 1)

        '''
        self.rewards += r

        self.s_0 = s_1.copy()

        self.step_count += 1
        if done:
            self.s_0 = self.env.reset()
        return done

    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=400,
              network_update_frequency=4,
              network_sync_frequency=8):
        self.gamma = gamma

        # for MAZE, show different codes before training
        #self.different_codes = verifyCodes(self.encoder, 5, 5, True, True)
        #showDistancesFromGoal(self.encoder, 5, 5, [4, 4])

        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
            print("explore")
        ep = 0
        training = True
        while training:
            self.s_0 = self.env.reset()
            self.rewards = 0
            done = False
            while done == False:
                if ((ep % 5) == 0):
                    self.env.render()
                #if ((ep % 10) == 0):
                    #self.different_codes = verifyCodes(self.encoder, 5, 5)
                    # verifyQ(self.network, 5, 5)
                    # showDistancesFromGoal(self.encoder, 5, 5, [4,4])
                    # save models
                    #self.save_models()

                p = np.random.random()
                if p < self.epsilon:
                    done = self.take_step(mode='explore')
                    # print("explore")
                else:
                    done = self.take_step(mode='train')
                    # print("train")
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())
                    self.sync_eps.append(ep)

                if done:
                    ep += 1
                    if self.epsilon >= 0.10:
                        self.epsilon -= 0.05
                    self.episode = ep
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    print(
                        "\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}   {} different codes \t\t".format(
                            ep, mean_rewards, self.rewards, self.different_codes), end="")
                    self.f.write(str(mean_rewards) + "\n")
                    #print("losses ", self.transition_losses[-self.window:])
                    # plot
                    '''
                    showDistancesFromGoal(self.encoder, 5, 5, [4,4])
                    plt.plot(self.transition_losses)
                    plt.title('Reaching goal [4,4]') # .format(self.env.goal[0],self.env.goal[1]))
                    plt.ylabel('Loss maze')
                    plt.xlabel('Episods')
                    plt.show('lossMaze.png') #.format(self.env.goal[0],self.env.goal[1]))
                    plt.clf()
                    '''
                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold and ep > 15:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        break
        # save models
        self.save_models()
        # plot
        '''
        showDistancesFromGoal(self.encoder, 5, 5, [4, 4])
        plt.plot(self.transition_losses)
        plt.title('Reaching goal [4,4]')  # .format(self.env.goal[0],self.env.goal[1]))
        plt.ylabel('Reward')
        plt.xlabel('Episods')
        plt.savefig('rewardMaze.png')  # .format(self.env.goal[0],self.env.goal[1]))
        plt.clf()
        '''

    def save_models(self):
        torch.save(self.encoder, self.path_to_the_models + "encoder")
        torch.save(self.decoder, self.path_to_the_models + "decoder")
        torch.save(self.trans_delta, self.path_to_the_models + "trans_delta")
        torch.save(self.network, self.path_to_the_models + "Q_net")

    def load_models(self):
        self.encoder = torch.load(self.path_to_the_models+"encoder")
        self.encoder.eval()
        self.decoder = torch.load(self.path_to_the_models+"decoder")
        self.decoder.eval()
        self.trans_delta = torch.load(self.path_to_the_models+"trans_delta")
        self.trans_delta.eval()
        self.network = torch.load(self.path_to_the_models+"Q_net")
        self.network.eval()

    def calculate_loss(self, batch):

        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1, 1)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(
            device=self.network.device)
        dones_t = torch.ByteTensor(dones).to(device=self.network.device)

        ###############
        # DDQN Update #
        ###############

        qvals = self.network.get_qvals(states)
        # qy = qy.to('cpu')
        qvals = torch.gather(qvals.to('cpu'), 1, actions_t)

        next_vals= self.network.get_qvals(next_states)
        next_actions = torch.max(next_vals.to('cpu'), dim=-1)[1]
        next_actions_t = torch.LongTensor(next_actions).reshape(-1, 1).to(
            device=self.network.device)
        target_qvals= self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals.to('cpu'), 1, next_actions_t).detach()
        ###############
        qvals_next[dones_t] = 0  # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t

        loss = (nn.MSELoss()(qvals, expected_qvals))

        # print("loss = ", loss)
        loss.backward()
        self.network.optimizer.step()

        return loss

    def pred_update(self, batch):
        loss_function = nn.MSELoss()
        states, actions, rewards, dones, next_states = [i for i in batch]
        cat_actions = []

        # modifica struttura actions
        for act in actions:
            cat_actions.append(np.asarray(to_categorical(act, self.action_size)))
        cat_actions = np.asarray(cat_actions)
        a_t = torch.FloatTensor(cat_actions).to(device)

        # Modifiche struttura states
        if type(states) is tuple:
            states = np.array([np.ravel(s) for s in states])
        states = torch.FloatTensor(states).to(device)

        # Modifiche struttura states
        if type(next_states) is tuple:
            next_states = np.array([np.ravel(s) for s in next_states])
        next_states = torch.FloatTensor(next_states).to(device)

        error_z, x_prime_hat = self.transition(states, a_t, next_states)
        L = self.transition.loss_function_transition(error_z, next_states, x_prime_hat)
        L.backward()
        self.transition_losses.append(L)
        self.transition.optimizer.step()
        return

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)

        # <-------------------------comment for Q-learning

        self.transition.optimizer.zero_grad()
        batch2 = self.buffer.sample_batch(batch_size=self.batch_size)
        self.pred_update(batch2)

        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self.env.reset()