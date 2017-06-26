# coding: utf-8

# In[1]:

import gym
import numpy as np
import cPickle as pickle
import tensorflow as tf

# env_d = 'LunarLander-v2'
env_d = 'CartPole-v0'

if env_d == 'CartPole-v0':
    total_episodes = 2e3
    obsrv_size = 4
    num_of_actions = 2
    n2 = 15
    n3 = 15
else:
    total_episodes = 3e4
    obsrv_size = 8
    num_of_actions = 3
    n2 = 15
    n3 = 15

batch_size = 10
learning_rate = 0.005
n_iterations = int(total_episodes // batch_size)

n1 = obsrv_size
n4 = num_of_actions
env = gym.make(env_d)

TRAIN = True
EVALUATE = True

# In[2]:


W1 = tf.get_variable(name="W1", shape=[n1, n2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name="W2", shape=[n2, n3], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable(name="W3", shape=[n3, n4], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name="B1", shape=[n2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name="B2", shape=[n3], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name="B3", shape=[n4], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
weight_list = [W1, b1, W2, b2, W3, b3]

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

observations_ph = tf.placeholder(tf.float32, [None, obsrv_size])
actions_ph = tf.placeholder(tf.float32, [None, num_of_actions])
rewards_ph = tf.placeholder(tf.float32, [None])

# In[3]:
def saveWeights(session, weight_list, fname):
    v_weight_list = session.run(weight_list)
    pickle.dump(v_weight_list, open(fname, 'wb'))

def loadWeights(session, weight_list, fname):
    v_weight_list = pickle.load(open(fname, 'rb'))
    for i in range(len(weight_list)):
        session.run(weight_list[i].assign(v_weight_list[i]))

def agent(observation):
    h1 = tf.nn.tanh(tf.matmul(observation, W1) + b1)
    h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
    return tf.nn.softmax(tf.matmul(h2, W3) + b3)

y = agent(observations_ph)
log_y = -tf.log(y)
acc_loss = tf.reduce_sum(actions_ph * log_y, 1) * rewards_ph
loss = tf.reduce_sum(acc_loss) / tf.constant(batch_size, dtype=tf.float32)
cost = opt.minimize(loss)

debug_dict = {
    'y': y,
    'log_y': log_y,
    'acc_loss': acc_loss,
    'loss': loss,
    'cost': cost
}


# In[4]:

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

if TRAIN:
    obsrv = env.reset()
    for e in range(n_iterations):
        print('-'*100)
        print('Iteration {}/{}'.format(e,n_iterations))
        print('-'*100)
        action_list = []
        obsrv_list  = []
        partial_sum_rewards_list = []
        reward_avg = 0.0
        for game_idx in range(batch_size):
            obsrv = env.reset() # Obtain an initial observation of the environment
            done = False
            reward_list = []
            while not done:
                action_probs = sess.run(y, feed_dict={observations_ph: obsrv.reshape([1, obsrv_size])})
                action_probs = action_probs.reshape(-1)
                action = np.argmax(np.random.multinomial(1, action_probs))
                obsrv_list.append(obsrv) 
                obsrv, reward, done, _ = env.step(action)
                # print('Action Probs: {}, Selected Action {}'.format(action_probs, action))
                # print('obsrv {}, reward {}, done {}\n'.format(obsrv, reward, done))
                reward_list.append(reward)
                reward_avg += reward

                action_vec = np.zeros_like(action_probs)
                action_vec[action] = 1 # we need to make it one-hot
                action_list.append(action_vec)

            # comulated_rewards = [np.sum(reward_list[i:]) for i in range(len(reward_list))]
            # partial_sum_rewards_list.extend(comulated_rewards)                
            for i in range(len(reward_list)):
                partial_sum_rewards_list.append(np.sum(reward_list[i:]))

        # run the optimization after the batch is over.
        v_debug_dict = sess.run(debug_dict, feed_dict = {
                            observations_ph: np.array(obsrv_list),
                            actions_ph: np.array(action_list),
                            rewards_ph: np.array(partial_sum_rewards_list)
            })
        print('Iteration {}, Loss {}, Average Reward {}'.format(e, v_debug_dict['loss'], reward_avg/batch_size))
        fname = 'weights/ws_{}.p'.format(e)
        print('Saving Weights')
        saveWeights(sess, weight_list, fname)
        print('Loading Weights')
        loadWeights(sess, weight_list, fname)
# In[7]:


# LET's PLAY!
# done = False
if EVALUATE:
    e = 150
    fname = 'weights/weights_{}.pkl'.format(e)
    loadWeights(sess, weight_list, fname)
    env = gym.make(env_d)
    reward_sum = 0.0
    for g in range(5):
        obsrv = env.reset() # Obtain an initial observation of the environment
        done = False
        print('Game {}'.format(g))
        for t in range(200):
            env.render()
            action_probs = sess.run(y, feed_dict={observations_ph: obsrv.reshape([1, obsrv_size])})
            action_probs = action_probs.reshape(-1)
            act = np.argmax(action_probs)
            obsrv, reward, done, info = env.step(act)
            reward_sum += reward
            print('Action Probs: {}, Selected Action {}'.format(action_probs, act))
            print('obsrv {}, reward {}, done {}\n'.format(obsrv, reward, done))
            # print(obsrv, reward, done, info)
            if done:
                print('-'*100)
                print("Episode finished after {} timesteps with reward {}".format(t+1, reward_sum))
                print('-'*100)
                reward_sum = .0
                done = False
                break
# sess.close()
# tf.reset_default_graph()