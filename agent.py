import tensorflow as tf
from utils import *
from huber_loss import *
from network import *
import numpy as np
import os

EPSILON_BEGIN = 1.0
EPSILON_END = 0.1
EPSILLON_DECAY_FACTOR = 0.999
BETA_BEGIN = 0.5
BETA_END = 1.0

def get_fixed_samples(env, num_actions, num_samples):
    fixed_samples = []
    num_environment = env.num_process
    env.reset()

    for _ in range(0, num_samples, num_environment):
        old_state, action, reward, new_state, is_terminal = env.get_state()
        action = np.random.randint(0, num_actions, size=(num_environment,))
        env.take_action(action)
        for state in new_state:
            fixed_samples.append(state)
    return np.array(fixed_samples)



def get_multi_step_sample(memory, gamma, num_step):
    """
    Args
    memory: list of (s, a, r, next_s, d)
        only the last element's d should be 1.
    gamma:  the discount factor
    num_step: num_step

    ---
    Return:
    """
    states = []
    actions = []
    accum_rewards = []
    new_states = []
    terminals = []
    for i in range(len(memory)):
        state = memory[i][0]
        action = memory[i][1]
        accum_r = 0
        terminal = 0
        if (i + num_step) >= len(memory):
            new_state = memory[-1][3]

        else:
            new_state = memory[i + num_step][0]

        for j in range(num_step):

            if i + j >= len(memory):
                break
            accum_r += gamma ** (j + 1) * memory[i + j][2]
            terminal += memory[i + j][4]

        states.append(state)
        actions.append(action)
        accum_rewards.append(accum_r)
        new_states.append(new_state)
        terminals.append(terminal)
    return states, actions, accum_rewards, new_states, terminals



def create_dqn(x, action_size, duel=True, nn_type='mlp', **kargs):
    if duel:
        q = duel_q_network(x, action_size, nn_type=nn_type, **kargs)

    else:
        q = q_network(x, action_size, nn_type=nn_type, **kargs)

    a = tf.argmax(q, axis=1)

    return q, a


def create_distributional_dqn(x, action_size, duel=True, nn_type='mlp', **kargs):
    N_atoms = 51
    V_max = 20.0
    V_min = 0.0
    delta_z = (V_max - V_min) / (N_atoms - 1)
    z_list = tf.constant([V_min + i * delta_z for i in range(N_atoms)])
    z_list_broadcasted = tf.tile(tf.reshape(z_list, [1, N_atoms]), tf.constant([action_size, 1]))

    if duel:
        q_distri = duel_q_network(x, action_size * N_atoms, nn_type=nn_type, **kargs)
        q_distri = tf.reshape(q_distri, [-1, action_size, N_atoms])
        # [batch_size, action_size, N_atoms]
        q_distri = tf.nn.softmax(q_distri, dim=2)
        # Clipping to prevent Nan
        q_distri = tf.clip_by_value(q_distri, 1e-8, 1.0 - 1e-8)

        q = tf.reduce_sum(q_distri * z_list_broadcasted, axis=2, name='q_values')
        mean_max_q = tf.reduce_mean(tf.reduce_max(q, axis=1), name='mean_max_q')
        a = tf.argmax(q, axis=1)


    else:
        q_distri = q_network(x, action_size * N_atoms, nn_type=nn_type, **kargs)
        q_distri = tf.reshape(q_distri, [-1, action_size, N_atoms])
        # [batch_size, action_size, N_atoms]
        q_distri = tf.nn.softmax(q_distri, dim=2)
        # Clipping to prevent Nan
        q_distri = tf.clip_py_value(q_distri, 1e-8, 1.0 - 1e-8)

        q = tf.reduce_sum(q_distri * z_list_broadcasted, axis=2, name='q_values')
        mean_max_q = tf.reduce_mean(tf.reduce_max(q, axis=1), name='mean_max_q')
        a = tf.argmax(q, axis=1)

    return q_distri, q, mean_max_q, a


class Agent(object):

    def __init__(self, state_size, action_size,
                 replay_memory,
                 dir = "dqn",
                 gamma=0.99,
                 update_freq=4,
                 target_update_freq=10000,
                 batch_size=128,
                 nn_type='mlp',
                 duel=True,
                 distributional=False,
                 double=True,
                 num_step=1,
                 per=True,
                 learning_rate=0.00025,
                 polyak=0.0
                 ):
        """
        Args
         state_size:
         action_size:
         gamma:
         update_freq:
         target_update_freq:
         batch_size:
         epsilon:
         nn_type:

         duel:
         distributional:
         double:
         num_step:
         per:
         learning_rate:
         polyak:
        """

        assert type(state_size) == list
        self.state_size = state_size
        self.dim = len(state_size)

        self.action_size = action_size
        self.replay_memory = replay_memory
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        #######
        self.gamma = gamma
        self.nn_type = nn_type
        #######
        self.duel = duel
        self.distributional = distributional
        self.double = double
        self.num_step = num_step
        self.per = per

        #######
        self.epsilon = EPSILON_BEGIN
        self.epsilon_increment = (EPSILON_END - EPSILON_BEGIN) / 2000000.0
        self.beta = BETA_BEGIN
        self.beta_increment = (EPSILON_END - BETA_BEGIN) / 2000000.0
        self.learning_rate = learning_rate
        self.polyak = polyak
        #######

        self.update_time = 0
        self.global_step = 0
        self.path = os.path.join("model", dir)
        """
        Distributional dqn graph build 
        """
        if self.distributional:
            self._distributional_build_graph()
        else:
            self._build_graph()

        self._init_session()
        self.target_update_op()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():

            self._placeholders()

            with tf.variable_scope('main'):
                self.main_q, self.a = create_dqn(self.s, self.action_size, duel=self.duel, nn_type=self.nn_type)

            with tf.variable_scope('target'):
                self.target_q, _ = create_dqn(self.target_s, self.action_size, duel=self.duel, nn_type=self.nn_type)

            """
            self.main_q, self.target_q
            self.a_for_new_state_ph: the action which has the maximun q in the given new state.
            and list type as (idx, action) for tf.gather_nd

            """
            if self.double:
                max_q = tf.gather_nd(self.target_q, self.a_for_new_state_ph)

            else:
                max_q = tf.reduce_max(self.target_q, axis=1)

            target = self.r_ph + (1.0 - self.d_ph) * (self.gamma ** self.num_step) * max_q

            output = tf.gather_nd(self.main_q, self.a_ph, name='estimated_output')

            if self.per:
                self.loss = weighted_huber_loss(target, output, self.loss_weight_ph)

            else:
                self.loss = mean_huber_loss(target, output)

            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = opt.minimize(self.loss, var_list=get_vars('main'))
            self.error_op = tf.abs(output - target, name='abs_error')



            self.target_update = tf.group([tf.assign(q_target, self.polyak * q_target + (1 - self.polyak) * q_main)
                                           for q_main, q_target in zip(get_vars('main'), get_vars('target'))])

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def _distributional_build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            with tf.variable_scope('main'):
                self.main_distri_prob_q, self.main_q, self.mean_max_q, self.a = \
                    create_distributional_dqn(self.s,
                                              self.action_size,
                                              duel=self.duel,
                                              nn_type=self.nn_type)
            with tf.variable_scope('target'):
                self.target_distri_prob_q, self.target_q, _, _ = \
                    create_distributional_dqn(self.target_s,
                                              self.action_size,
                                              duel=self.duel,
                                              nn_type=self.nn_type)


            N_atoms = 51
            V_max = 20.0
            V_min = 0.0
            delta_z = (V_max - V_min) / (N_atoms - 1)
            z_list = tf.constant([V_min + i * delta_z for i in range(N_atoms)])
            z_list_broadcasted = tf.tile(tf.reshape(z_list, [1, N_atoms]), tf.constant([self.action_size, 1]))
            tmp_batch_size = tf.shape(self.main_q)[0]

            if self.duel:
                # batch_size * N_atoms
                prob_q_chosen_by_action_target = tf.gather_nd(self.target_distri_prob_q, self.a_for_new_state_ph)


            else:
                action_chosen_by_target = tf.cast(tf.argmax(self.target_q, axis=1), tf.int32)
                prob_q_chosen_by_action_target = tf.gather_nd(self.target_distri_prob_q,
                    tf.concat(tf.reshape(tf.range(tmp_batch_size), [-1, 1]), action_chosen_by_target, axis=1))

            # batch_size* N_atoms
            target = tf.tile(tf.reshape(self.r_ph, [-1, 1]), tf.constant([1, N_atoms]))
            target += (self.gamma ** self.num_step) * tf.reshape(z_list, [1, N_atoms]) \
                    * (1 - tf.tile(tf.reshape(self.d_ph, [-1, 1]), tf.constant([1, N_atoms])))
            target = tf.clip_by_value(target, V_min, V_max)

            #Indexes
            # max b = N_atoms - 1, index 0 ~ 50
            b = (target - V_min) / delta_z
            # u + l == 1
            u, l = tf.ceil(b), tf.floor(b)
            u_id, l_id = tf.cast(u, tf.int32), tf.cast(l, tf.int32)


            """
            Indexing for tf.gather_nd
            """
            # batch_size * N_atoms  * 1
            index = tf.expand_dims(tf.tile(tf.reshape(tf.range(tmp_batch_size), [-1, 1]), tf.constant([1, N_atoms])),-1)

            # u_index[idx in bathsize] = [[the index of batch, the chosen index in that N_atoms]] * N_atoms
            u_index = tf.concat([index, tf.expand_dims(u_id, -1)], axis=2)
            l_index = tf.concat([index, tf.expand_dims(l_id, -1)], axis=2)
            prob_q_chosen_by_action_main = tf.gather_nd(self.main_distri_prob_q, self.a_ph)

            # cross entropy
            # batch_size * N_atoms
            cross_entropy = (u - b) * prob_q_chosen_by_action_target * tf.log(tf.gather_nd(prob_q_chosen_by_action_main, u_index)) \
                            + (b - l) * prob_q_chosen_by_action_target * tf.log(tf.gather_nd(prob_q_chosen_by_action_main, l_index))

            # batch_size
            error = tf.reduce_sum(cross_entropy, axis=1)
            if self.per:
                self.loss = -1 * error * self.loss_weight_ph

            else:
              self.loss = -1 * error

            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = opt.minimize(self.loss, var_list=get_vars("main"))
            self.error_op = tf.abs(error, name='abs_errror')

            self.target_update = tf.group([tf.assign(q_targ, q_targ * self.polyak + q_main*(1 - self.polyak))
                                        for q_main, q_targ in zip(get_vars('main'), get_vars('target'))])

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()


    def _placeholders(self):
        self.s = tf.placeholder(shape=[None] + self.state_size, dtype=tf.float32, name='s')
        self.target_s = tf.placeholder(shape=[None] + self.state_size, dtype=tf.float32, name='target_s')
        self.a_ph = tf.placeholder(shape=(None, 2), dtype=tf.int32, name='a_ph')
        self.r_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='r_ph')
        self.d_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='d_ph')
        self.a_for_new_state_ph = tf.placeholder(shape=(None, 2), dtype=tf.int32, name='a_for_new_state_ph')
        self.loss_weight_ph = tf.placeholder(tf.float32, name='loss_weight_ph')

        ############################################################################

    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.summary_writer = tf.summary.FileWriter("logs/", graph=self.g)
        self.sess.run(self.init)

    def get_action(self, s, epsilon):
        """
        Args
         s: the current states ,ex (num_env, w, h, num_frames)

        Return
         a: int for the action within (0, action_size-1)

        """
        batch_size = len(s)
        if np.random.rand() < epsilon:
            action = np.random.randint(self.action_size, size=(batch_size,))

        else:

            feed_dict = {self.s: s/255.}
            action = self.sess.run(self.a, feed_dict=feed_dict)

        return action

    def get_multi_step_sample(self, env):
        old_state, action, reward, new_state, is_terminal = env.get_state()
        # Clip the reward to -1, 0, 1
        total_reward = np.sign(reward)
        total_is_terminal = is_terminal
        next_action = self.get_action(new_state, self.epsilon)
        env.take_action(next_action)
        #print("num_step::::".format(self.num_step))
        for i in range(1, self.num_step):
            #print(i)
            _ , _ , reward, new_state, is_terminal = env.get_state()
            # Clip the reward to -1, 0, 1
            total_reward = total_reward + self.gamma**i * np.sign(reward)
            total_is_terminal = total_is_terminal + is_terminal
            next_action = self.get_action(new_state, self.epsilon)
            env.take_action(next_action)

        return old_state, action, total_reward, new_state, np.sign(total_is_terminal)

    def fit(self, env, num_iteration, do_train=False):

        #s, a, r, new_s, d = get_multi_step_sample(one_step_memory, self.gamma, self.num_step)
        #self.replay_memory.append((s, a, r, new_s, d))
        # epsilon update
        """
        Epsilon update
        epsilon begin 1.0, end up 0.1
        FIX
        """
        num_env = env.num_process
        env.reset()

        for t in range(0, num_iteration, num_env):
            self.global_step += 1
            #print("Global_step: {}".format(self.global_step))
            old_state, action, reward, new_state, is_terminal = self.get_multi_step_sample(env)
            self.replay_memory.append(old_state, action, reward, new_state, is_terminal)

            """
                    Epsilon update
                    epsilon begin 1.0, end up 0.1
                    FIX
            """

            self.epsilon = self.epsilon+ num_env*self.epsilon_increment if self.epsilon > EPSILON_END else EPSILON_END
            num_update = sum([1 if i%self.update_freq == 0 else 0 for i in range(t, t+num_env)])
            if do_train:
                for _ in range(num_update):

                    if self.per == 1:
                        (old_state_list, action_list, reward_list, new_state_list, is_terminal_list), \
                        idx_list, p_list, sum_p, count = self.replay_memory.sample(self.batch_size)
                    else:
                        old_state_list, action_list, reward_list, new_state_list, is_terminal_list \
                            = self.replay_memory.sample(self.batch_size)

                    feed_dict = {self.target_s: new_state_list.astype(np.float32)/255. ,
                             self.s : old_state_list.astype(np.float32)/255.,
                             self.a_ph: list(enumerate(action_list)),
                             self.r_ph: np.array(reward_list).astype(np.float32),
                             self.d_ph: np.array(is_terminal_list).astype(np.float32),
                             }

                    if self.double:
                        action_chosen_by_online = self.sess.run(self.a,
                                                                feed_dict={
                                                                    self.s: new_state_list.astype(np.float32)/255.})
                        feed_dict[self.a_for_new_state_ph] = list(enumerate(action_chosen_by_online))

                    if self.per == 1:
                        # Annealing weight beta
                        feed_dict[self.loss_weight_ph] = (np.array(p_list) * count / sum_p) ** (-self.beta)
                        error, _ = self.sess.run([self.error_op, self.train_op], feed_dict=feed_dict)
                        self.replay_memory.update(idx_list, error)

                    else:
                        self.sess.run(self.train_op, feed_dict=feed_dict)

                    self.update_time += 1

                    if self.beta < BETA_END:
                        self.beta += self.beta_increment

                    if (self.update_time)%self.target_update_freq == 0 :
                        #print("Step: {} ".format(self.update_time) + "target_network update")
                        self.sess.run([self.target_update])
                        #print("Step: {} ".format(self.update_freq) + "Network save")
                        self.save_model()


    def target_update_op(self):
        self.sess.run([self.target_update])
        #print("Target Model UPDATE")

    def evaluate(self, env, num_episode, epsilon):
        """Evaluate num_episode games by online model.
        Parameters
        ----------
        sess: tf.Session
        env: batchEnv.BatchEnvironment
          This is your paralleled Atari environment.
        num_episode: int
          This is the number of episode of games to evaluate
        Returns
        -------
        reward list for each episode
        """
        num_environment = env.num_process
        env.reset()
        reward_of_each_environment  = np.zeros(num_environment)
        rewards_list = []

        num_finished_episode = 0

        while num_finished_episode < num_episode:
            old_state, action, reward, new_state, is_terminal = env.get_state()
            action = self.get_action(new_state, epsilon)
            env.take_action(action)
            for i, r, is_t in zip(range(num_environment), reward, is_terminal):
                if not is_t:
                    reward_of_each_environment[i] += r
                else:
                    rewards_list.append(reward_of_each_environment[i])
                    reward_of_each_environment[i] = 0
                    num_finished_episode += 1
        return np.mean(rewards_list), np.std(rewards_list), self.epsilon

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def save_model(self):
        self.saver.save(self.sess, self.path + '/q_network', global_step=self.update_time)

    def _is_model_exists(self):
        exists = tf.train.latest_checkpoint(self.path)
        if exists is not None:
            return True

        else:
            return False

    def load_model(self):
        if self._is_model_exists():
            latest_ckpt = tf.train.latest_checkpoint(self.path)
            self.saver.restore(self.sess, latest_ckpt)
            return True

        else:
            return False





if __name__ == "__main__":
    Agent(10, 5)