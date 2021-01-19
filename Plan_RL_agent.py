import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from QNetwork import QNetwork

from Autoencoder import Encoder, Decoder
# from testExplainability import verifyCodes, showDistancesFromGoal
from Node import Node
from system_conf import ACTION_SIZE, CODE_SIZE, Q_HIDDEN_NODES, BATCH_SIZE, REW_THRE, WINDOW, MODELS_DIR, MARGIN
from transition_model import Transition, TransitionDelta
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class Plan_RL_agent:

    def __init__(self, env, buffer, load_models = False, epsilon=0.5, Q_hidden_nodes = Q_HIDDEN_NODES, batch_size= BATCH_SIZE, rew_thre = REW_THRE, window = WINDOW, path_to_the_models = MODELS_DIR):

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
        _, x_prime = self.trans_delta(x, torch.from_numpy(a).type(torch.FloatTensor).to(device))
        return x_prime

    def vFunc(self, x):
        v0 = self.network.get_enc_value(x)
        return torch.max(v0).to('cpu').detach().numpy()

    def certainty(self, x):
        x_p = self.encoder(self.decoder(x))
        distance = torch.nn.L1Loss()
        c = 1 - distance(x, x_p).item()
        return c


    def findPlan(self, node):
        # caso base
        if node.sons == []:
            return [node.a], node.v*node.c

        somme_values = []
        plans = []
        for n in node.sons:
            p, s = self.findPlan(n)
            plans.append(p)
            somme_values.append(s)
            # print("plan p", p)
            # print("plan p", s)

        ###### evaluate plans
        #se più piani hanno valore massimo ne scelgo uno fra di essi random
        smax = max(somme_values)
        indices_max = [i for i, j in enumerate(somme_values) if j == smax]
        k = random.choice(indices_max)

        bestp = plans[k]

        return [node.a] + bestp, node.v * node.c + smax

    def limited_expansion(self, node, depth):
        if depth == 0:
            return

        for a in self.A:
            x_prime = self.expandFunc(node.x, a)

            node.expand(x_prime, self.vFunc(x_prime), a, node.c * self.certainty(x_prime))

        for i in range(len(node.sons)):
            self.limited_expansion(node.sons[i], depth - 1)

    def planner_action(self, depth=2, verbose = False):
        #if np.random.random() < 0.05:
        #    return np.random.choice(self.action_size)
        origin_code = self.encoder(torch.from_numpy(self.s_0).type(torch.FloatTensor))
        origin_value = self.vFunc(origin_code)
        root = Node(origin_code, origin_value, to_categorical(0, self.action_size), self.certainty(origin_code))

        self.limited_expansion(root, depth)

        plan, sum_value = self.findPlan(root)

        if verbose:
            root.print_parentetic()
            print("plan: {}, sum_value: {}".format(plan[1:], sum_value))

        return np.where(plan[1] == 1)[0][0]

    def is_diff(self, s1, s0):
        for i in range(len(s0)):
            if (s0[i] != s1[i]):
                return True
        return False

    def take_step(self, mode='train', horizon = 0):

        #actions = ['N', 'E', 'S', 'W']                              # <-----decomment for maze
        #s_1, r, done, _ = self.env.step(actions[self.action])       #
        s, r, done, _ = self.env.step(self.action)

        #TODO: sistemare questa parte
        '''
        enc_s1 = self.encoder(torch.from_numpy(np.asarray(s_1)).type(torch.FloatTensor))
        enc_s0 = self.encoder(torch.from_numpy(np.asarray(self.s_0)).type(torch.FloatTensor).to('cuda'))
        # print("Reward = ", r)
        if(self.is_diff(enc_s0,enc_s1)):
        # print("step passati = ", self.step_count - self.timestamp)
        # self.timestamp = self.step_count
        '''
        #self.buffer.append(self.s_0, self.action, r, done, s_1)

        if self.a_1 != -1:
            #                  (  s_0     , a_1   ,   r_1    ,    d_1    ,    s_1    ,    a_2  ,r_2, d_2,s_2)
            self.buffer.append(self.s_0, self.a_1, self.r_1, self.done_1, self.s_1, self.action, r, done, s )
            self.s_0 = self.s_1.copy()
        self.a_1 = self.action
        self.r_1 = r
        self.done_1 = done
        self.s_1 = s


        # self.cum_rew = 0

        if mode == 'explore':
            self.action = self.env.action_space.sample()
        else:
            #self.action = self.network.get_action(self.s_0)

            if horizon != 0:
              self.action = self.planner_action(depth=horizon)
            else:
                # ADAPTIVE HORIZON
                if len(self.mean_training_rewards) < 1: #<---------le threshold sono per il maze
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
                                    self.action = self.planner_action(depth= 2)
                                else:
                                    self.planner_action(depth= 3)

        self.rewards += r


        self.step_count += 1
        if done:
            self.s_0 = self.env.reset()
            self.a_1 = -1
        return done

    def monitor_replanning(self, horizon):
        done = False
        while not done:
            done = self.take_step(horizon = horizon)


    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=4,
              network_update_frequency=4,
              network_sync_frequency=200):
        self.gamma = gamma

        # for MAZE, show different codes before training
        #self.different_codes = verifyCodes(self.encoder, 5, 5, True, True)
        #showDistancesFromGoal(self.encoder, 5, 5, [4, 4])

        self.s_0 = self.env.reset()
        self.a_1 = -1
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
            #print("explore")
        ep = 0
        training = True
        while training:
            self.s_0 = self.env.reset()
            self.a_1 = -1
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
                    done = self.take_step(mode='train', horizon=1)
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

                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold and ep > 15:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        #break
        # save models
        self.save_models()
        # plot
        self.plot_training_rewards()

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

    def plot_training_rewards(self):
        plt.plot(self.mean_training_rewards)
        plt.title('Mean training rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episods')
        plt.show()
        plt.savefig(self.path_to_the_models+'mean_training_rewards.png')
        plt.clf()

    def calculate_loss(self, batch):

        states, actions, rewards, dones, next_states, _, _, _, _ = [i for i in batch]
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
        states, actions, rewards, dones, next_states, _, _, _, _ = [i for i in batch]
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

        L = self.transition.one_step_loss(states, a_t, next_states)
        L.backward()
        #self.transition_losses.append(L)
        self.transition.optimizer.step()
        return


    def pred_update_two_steps(self, batch):
        loss_function = nn.MSELoss()
        states, actions, rewards, dones, next_states, actions_2, rewards_2, dones_2, next_states_2 = [i for i in batch]
        cat_actions = []
        cat_actions_2 = []

        # modifica struttura actions
        for act in actions:
            cat_actions.append(np.asarray(to_categorical(act, self.action_size)))
        cat_actions = np.asarray(cat_actions)
        a_t = torch.FloatTensor(cat_actions).to(device)

        # modifica struttura actions_2
        for act in actions:
            cat_actions_2.append(np.asarray(to_categorical(act, self.action_size)))
        cat_actions_2 = np.asarray(cat_actions)
        a_t_2 = torch.FloatTensor(cat_actions_2).to(device)

        # Modifiche struttura states
        if type(states) is tuple:
            states = np.array([np.ravel(s) for s in states])
        states = torch.FloatTensor(states).to(device)

        # Modifiche struttura next_states
        if type(next_states) is tuple:
            next_states = np.array([np.ravel(s) for s in next_states])
        next_states = torch.FloatTensor(next_states).to(device)

        # Modifiche struttura next_states
        if type(next_states_2) is tuple:
            next_states_2 = np.array([np.ravel(s) for s in next_states_2])
        next_states_2 = torch.FloatTensor(next_states_2).to(device)

        L = self.transition.two_step_loss(states, a_t, next_states, a_t_2, next_states_2)

        #se mettiamo pure la triplet loss
        #L + self.transition.triplet_loss_encoder(states, next_states, next_states_2, MARGIN)

        L.backward()
        #self.transition_losses.append(L)

        self.transition.optimizer.step()
        return

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)

        # <-------------------------comment for Q-learning

        self.transition.optimizer.zero_grad()
        batch2 = self.buffer.sample_batch(batch_size=self.batch_size)
        self.pred_update_two_steps(batch2)

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