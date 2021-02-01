import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from AutoEncoder import Encoder, Decoder
from QNetwork import QNetwork
from transition_model import Transition, TransitionDelta
from Node import Node
import random
import itertools

from system_conf import STATE_SIZE, ACTION_SIZE, CODE_SIZE, Q_HIDDEN_NODES, BATCH_SIZE, REW_THRE, WINDOW, MODELS_DIR, MARGIN, PREDICT_CERTAINTY
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class plan_node():
    def __init__(self, code_state, value):
        self.code_state = code_state
        self.value = value
        self.action_vec = []

    def add_action(self, a):
        self.action_vec.extend(a)


class Plan_RL_agent:
    def __init__(self, env, buffer, load_models = False, epsilon=0.05, Q_hidden_nodes = Q_HIDDEN_NODES, batch_size= BATCH_SIZE, rew_thre = REW_THRE, window = WINDOW, path_to_the_models = MODELS_DIR):

        print("MARGIN: ", MARGIN)
        print(("1/MARGIN: ", 1/MARGIN))
        self.lq = 0
        self.lts = 0
        self.ltx = 0
        self.ld = 0
        self.path_to_the_models = path_to_the_models
        self.env = env

        self.action_size = ACTION_SIZE
        self.state_size = STATE_SIZE
        self.code_size = CODE_SIZE

        if load_models:
            self.load_models()
        else:
            self.encoder = Encoder(self.code_size)
            self.decoder = Decoder(self.code_size)
            self.trans_delta = TransitionDelta(self.code_size, self.action_size)
            self.network = QNetwork(env=env, n_hidden_nodes=Q_hidden_nodes, encoder=self.encoder)

        self.transition = Transition(self.encoder, self.decoder, self.trans_delta)
        params = [self.encoder.parameters(),self.decoder.parameters(), self.trans_delta.parameters(), self.network.symbolic_net.parameters()]
        params = itertools.chain(*params)
        self.optimizer = torch.optim.Adam(params,
                                         lr=0.001)
        #self.f = open("res/planner_enc_DDQN.txt", "a+")
        self.target_network = deepcopy(self.network)
        self.buffer = buffer
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.window = window
        self.reward_threshold = rew_thre
        self.initialize()
        self.action = 0
        self.temp_s1 = 0
        self.step_count = 0
        self.cum_rew = 0
        self.timestamp = 0
        self.episode = 0
        self.difference = 0
        self.A = [to_categorical(i, self.action_size) for i in range(self.action_size)]


    def monitor_replanning(self, horizon, show = True):
        done = False
        self.rewards = 0
        while not done:
            if show:
                self.env.render()
            done = self.take_step(horizon = horizon)
        if show:
            print("Episode reward: ", self.rewards)
        return self.rewards

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
        #plt.show()
        plt.savefig(self.path_to_the_models+'mean_training_rewards.png')
        plt.clf()

    def expandFunc(self, x, a):
        _, x_prime = self.trans_delta(x, torch.from_numpy(a).type(torch.FloatTensor).to(device))
        return x_prime

    def vFunc(self, x):
        v0 = self.network.get_enc_value(x)
        return torch.max(v0).to('cpu').detach().numpy()

    def certainty(self, x):
        if PREDICT_CERTAINTY:
            x_p = self.encoder(self.decoder(x))
            distance = torch.nn.L1Loss()
            c = 1 - distance(x, x_p).item()
        else:
            return 1
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

    def planner_action(self, depth=1, verbose = False):
        if np.random.random() < 0.05:
            return np.random.choice(self.action_size)

        origin_code = self.encoder(torch.from_numpy(self.s_0).type(torch.FloatTensor))
        #print("Origin code: ", origin_code)
        origin_value = self.vFunc(origin_code)
        root = Node(origin_code, origin_value, to_categorical(0, self.action_size), self.certainty(origin_code))

        self.limited_expansion(root, depth)

        if verbose:
            root.print_parentetic()

        plan, sum_value = self.findPlan(root)

        if verbose:
            #root.print_parentetic()
            print("plan: {}, sum_value: {}".format(plan[1:], sum_value))

        return np.where(plan[1] == 1)[0][0]

    def planner_action_old(self, depth=1):
        #if np.random.random() < 0.05:
        #    return np.random.choice(self.action_size)

        origin_code = self.encoder(torch.from_numpy(self.s_0).type(torch.FloatTensor))
        origin_value = self.network.get_enc_value(origin_code)
        origin_node = plan_node(origin_code, origin_value)
        origin_node.action_vec = [0]
        action = torch.argmax(origin_value).to('cpu').detach().numpy()

        a0 = to_categorical(0,self.action_size)
        a1 = to_categorical(1,self.action_size)
        a2 = to_categorical(2,self.action_size)
        #a3 = to_categorical(3,self.action_size)
        #a4 = to_categorical(3, self.action_size)
        #a5 = to_categorical(3, self.action_size)


        _, ns0 = self.trans_delta(origin_code, torch.from_numpy(a0).type(torch.FloatTensor).to('cuda'))
        _, ns1 = self.trans_delta(origin_code, torch.from_numpy(a1).type(torch.FloatTensor).to('cuda'))
        _, ns2 = self.trans_delta(origin_code, torch.from_numpy(a2).type(torch.FloatTensor).to('cuda'))
        #_, ns3 = self.trans_delta(origin_code, torch.from_numpy(a3).type(torch.FloatTensor).to('cuda'))
        #_, ns4 = self.trans_delta(origin_code, torch.from_numpy(a4).type(torch.FloatTensor).to('cuda'))
        #_, ns5 = self.trans_delta(origin_code, torch.from_numpy(a5).type(torch.FloatTensor).to('cuda'))

        v0 = self.network.get_enc_value(ns0)
        v1 = self.network.get_enc_value(ns1)
        v2 = self.network.get_enc_value(ns2)
        #v3 = self.network.get_enc_value(ns3)
        #v4 = self.network.get_enc_value(ns4)
        #v5 = self.network.get_enc_value(ns5)

        max0 = torch.max(v0).to('cpu').detach().numpy()
        arg_max0 = torch.argmax(v0).to('cpu').detach().numpy()

        max1 = torch.max(v1).to('cpu').detach().numpy()
        arg_max1 = torch.argmax(v1).to('cpu').detach().numpy()


        max2 = torch.max(v2).to('cpu').detach().numpy()
        arg_max2 = torch.argmax(v2).to('cpu').detach().numpy()

        '''
        max3 = torch.max(v3).to('cpu').detach().numpy()
        arg_max3 = torch.argmax(v3).to('cpu').detach().numpy()

        
        max4 = torch.max(v4).to('cpu').detach().numpy()
        arg_max4 = torch.argmax(v4).to('cpu').detach().numpy()

        max5 = torch.max(v5).to('cpu').detach().numpy()
        arg_max5 = torch.argmax(v5).to('cpu').detach().numpy()
        '''

        l_max = [max0, max1, max2]

        #smax = max(l_max)
        #indices_max = [i for i, j in enumerate(l_max) if j == smax]
        #k = random.choice(indices_max)

        #l_amax = [arg_max0, arg_max1, arg_max2]
        l_amax = [0, 1, 2]

        #if(action != l_amax[np.argmax(l_max)]):
            #print("DIVERSO!")

        #return k
        return l_amax[np.argmax(l_max)]

    def is_diff(self, s1, s0):
        for i in range(len(s0)):
            if(s0[i] != s1[i]):
                return True
        return False

    def take_step(self, mode='train', horizon=1):

        s_1, r, done, _ = self.env.step(self.action)
        #print(self.env.action_space)
        enc_s1 = self.encoder(torch.from_numpy(np.asarray(s_1)).type(torch.FloatTensor))
        enc_s0 = self.encoder(torch.from_numpy(np.asarray(self.s_0)).type(torch.FloatTensor).to('cuda'))
        #print("Reward = ", r)
        if(self.is_diff(enc_s0,enc_s1)):
        #if(True):
            #print("step passati = ", self.step_count - self.timestamp)
            self.timestamp = self.step_count

            self.buffer.append(self.s_0, self.action, r, done, s_1)
            self.cum_rew = 0

            if mode == 'explore':
                self.action = self.env.action_space.sample()

            else:
                #self.action = self.network.get_action(self.s_0)
                self.action = self.planner_action()
                # ADAPTIVE HORIZON
                '''
                if len(self.mean_training_rewards) < 1 or self.mean_training_rewards[-1] < -400:  # <---------le threshold sono per il maze
                    self.action = self.network.get_action(self.s_0)
                else:
                    self.action = self.planner_action(depth=1)  # comment for Q-learning
                '''
            self.s_0 = s_1.copy()

        self.rewards += r
        self.step_count += 1
        if done:

            self.s_0 = self.env.reset()
        return done

    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=500,
              network_update_frequency=4,
              network_sync_frequency=200):
        self.gamma = gamma
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')

        ep = 0
        training = True
        while training:
            self.s_0 = self.env.reset()
            self.rewards = 0
            done = False
            while done == False:
                if((ep % 20) == 0 ):
                    self.env.render()

                p = np.random.random()
                if p < self.epsilon:
                    done = self.take_step(mode='explore')
                    # print("explore")
                else:
                    done = self.take_step(mode='train', horizon=2)
                    # print("train")
                #done = self.take_step(mode='train')
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
                    if self.epsilon >= 0.05:
                        self.epsilon = self.epsilon * 0.7
                    self. episode = ep
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    print("\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}  lq = {:.3f}  lts ={:3f}  ltx ={:3f}  ld ={:3f}  l_spars={:3f}\t\t".format(
                        ep, mean_rewards, self.rewards, self.lq, self.lts, self.ltx, self.ld, self.l_spars ), end="")
                    #self.f.write(str(mean_rewards)+ "\n")


                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        break
        # save models
        self.save_models()
        # plot
        self.plot_training_rewards()
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
        qvals = torch.gather(qvals.to('cpu'), 1, actions_t)

        next_vals= self.network.get_qvals(next_states)
        next_actions = torch.max(next_vals.to('cpu'), dim=-1)[1]
        next_actions_t = torch.LongTensor(next_actions).reshape(-1, 1).to(
            device=self.network.device)
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals.to('cpu'), 1, next_actions_t).detach()
        ###############
        qvals_next[dones_t] = 0  # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t

        self.lq = (nn.MSELoss()(qvals, expected_qvals))

        #print("loss = ", loss)
        #loss.backward()
        #self.network.optimizer.step()

        return self.lq

    def pred_update(self, batch):
        loss_function = nn.MSELoss()
        states, actions, rewards, dones, next_states = [i for i in batch]
        cat_actions = []

        #modifica struttura actions
        for act in actions:
            cat_actions.append(np.asarray(to_categorical(act,self.action_size)))
        cat_actions = np.asarray(cat_actions)
        a_t = torch.FloatTensor(cat_actions).to('cuda')

        #Modifiche struttura states
        if type(states) is tuple:
            states = np.array([np.ravel(s) for s in states])
        states = torch.FloatTensor(states).to('cuda')

        # Modifiche struttura states
        if type(next_states) is tuple:
            next_states = np.array([np.ravel(s) for s in next_states])
        next_states = torch.FloatTensor(next_states).to('cuda')

        self.ltx, self.lts = self.transition.one_step_loss(states, a_t, next_states)
        # per renderla comparabile alla lq
        #self.ltx *= 50
        self.ld = self.transition.distant_codes_loss(states, next_states)
        self.l_spars = self.transition.smooth_discrete_loss(states)
        #self.l_spars += self.transition.sparsity_loss(next_states)
        L = self.lts + self.ltx + self.ld + self.l_spars
        #L.backward()
        #print("pred_loss = ", L)
        #self.transition.optimizer.step()

        return L

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

        ####### NEW
        L = self.transition.two_step_loss(states, a_t, next_states, a_t_2, next_states_2)
        #se mettiamo pure la triplet loss
        #L + self.transition.triplet_loss_encoder(states, next_states, next_states_2, MARGIN)



        L.backward()
        #self.transition_losses.append(L)

        self.transition.optimizer.step()
        return

    def update(self):
        #self.network.optimizer.zero_grad()
        self.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)

        loss_q = self.calculate_loss(batch)
        #print("q loss = ", loss)

        #self.transition.optimizer.zero_grad()
        batch2 = self.buffer.sample_batch(batch_size=self.batch_size)
        loss_t = self.pred_update(batch2)
        #TODO calcolare la loss su un batch solo
        #batch_cons = self.buffer.consecutive_sample(batch_size=64)
        #print(batch_cons)

        loss = loss_t + loss_q
        loss.backward()
        self.optimizer.step()
        '''
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())
        '''
    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self.env.reset()
