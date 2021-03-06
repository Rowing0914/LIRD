import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from ou_noise import OrnsteinUhlenbeckNoise
from actor import Actor
from critic import Critic
from replay_memory import ReplayMemory, experience_replay
from env import Environment
from embedding import Embeddings, EmbeddingsGenerator
from data_generator import DataGenerator


def read_file(data_path):
    ''' Load data from train.csv or test.csv. '''

    data = pd.read_csv(data_path, sep=';')
    for col in ['state', 'n_state', 'action_reward']:
        data[col] = [np.array([[np.int(k) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]
    for col in ['state', 'n_state']:
        data[col] = [np.array([e[0] for e in l]) for l in data[col]]

    data['action'] = [[e[0] for e in l] for l in data['action_reward']]
    data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]
    data.drop(columns=['action_reward'], inplace=True)

    return data


def read_embeddings(embeddings_path):
    ''' Load embeddings (a vector for each item). '''

    embeddings = pd.read_csv(embeddings_path, sep=';')

    return np.array([[np.float64(k) for k in e.split('|')]
                     for e in embeddings['vectors']])


# Training
def train(sess, environment, actor, critic, embeddings, history_length, ra_length, buffer_size, batch_size,
          discount_factor, nb_episodes, filename_summary):
    ''' Algorithm 3 in article. '''

    # Set up summary operators
    def build_summaries():
        episode_reward = tf.Variable(0.)
        tf.summary.scalar('reward', episode_reward)
        episode_max_Q = tf.Variable(0.)
        tf.summary.scalar('max_Q_value', episode_max_Q)
        critic_loss = tf.Variable(0.)
        tf.summary.scalar('critic_loss', critic_loss)

        summary_vars = [episode_reward, episode_max_Q, critic_loss]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(filename_summary, sess.graph)

    # '2: Initialize target network f??? and Q???'
    actor.init_target_network()
    critic.init_target_network()

    # '3: Initialize the capacity of replay memory D'
    replay_memory = ReplayMemory(buffer_size)  # Memory D in article
    replay = False

    start_time = time.time()
    for i_session in range(nb_episodes):  # '4: for session = 1, M do'
        session_reward = 0
        session_Q_value = 0
        session_critic_loss = 0

        # '5: Reset the item space I' is useless because unchanged.

        states = environment.reset()  # '6: Initialize state s_0 from previous sessions'

        if (i_session + 1) % 10 == 0:  # Update average parameters every 10 episodes
            environment.groups = environment.get_groups()

        exploration_noise = OrnsteinUhlenbeckNoise(history_length * embeddings.size())

        for t in range(nb_rounds):  # '7: for t = 1, T do'
            # '8: Stage 1: Transition Generating Stage'

            # '9: Select an action a_t = {a_t^1, ..., a_t^K} according to Algorithm 2'
            actions = actor.get_recommendation_list(
                ra_length,
                states.reshape(1, -1),  # TODO + exploration_noise.get().reshape(1, -1),
                embeddings).reshape(ra_length, embeddings.size())

            # '10: Execute action a_t and observe the reward list {r_t^1, ..., r_t^K} for each item in a_t'
            rewards, next_states = environment.step(actions)

            # '19: Store transition (s_t, a_t, r_t, s_t+1) in D'
            replay_memory.add(states.reshape(history_length * embeddings.size()),
                              actions.reshape(ra_length * embeddings.size()),
                              [rewards],
                              next_states.reshape(history_length * embeddings.size()))

            states = next_states  # '20: Set s_t = s_t+1'

            session_reward += rewards

            # '21: Stage 2: Parameter Updating Stage'
            if replay_memory.size() >= batch_size:  # Experience replay
                replay = True
                replay_Q_value, critic_loss = experience_replay(replay_memory, batch_size,
                                                                actor, critic, embeddings, ra_length,
                                                                history_length * embeddings.size(),
                                                                ra_length * embeddings.size(), discount_factor)
                session_Q_value += replay_Q_value
                session_critic_loss += critic_loss

            summary_str = sess.run(summary_ops,
                                   feed_dict={summary_vars[0]: session_reward,
                                              summary_vars[1]: session_Q_value,
                                              summary_vars[2]: session_critic_loss})

            writer.add_summary(summary_str, i_session)

            '''
            print(state_to_items(embeddings.embed(data['state'][0]), actor, ra_length, embeddings),
                  state_to_items(embeddings.embed(data['state'][0]), actor, ra_length, embeddings, True))
            '''

        str_loss = str('Loss=%0.4f' % session_critic_loss)
        print(('Episode %d/%d Reward=%d Time=%ds ' + (str_loss if replay else 'No replay')) % (
            i_session + 1, nb_episodes, session_reward, time.time() - start_time))
        start_time = time.time()

    writer.close()
    tf.train.Saver().save(sess, 'models.h5', write_meta_graph=False)


## Data

# Hyperparameters
history_length = 12  # N in article
ra_length = 4  # K in article
discount_factor = 0.99  # Gamma in Bellman equation
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001  # ?? in Algorithm 3
batch_size = 64
nb_episodes = 100
nb_rounds = 50
filename_summary = 'summary.txt'
alpha = 0.5  # ?? (alpha) in Equation (1)
gamma = 0.9  # ?? (Gamma) in Equation (4)
buffer_size = 1000000  # Size of replay memory D in article
fixed_length = True  # Fixed memory length

dg = DataGenerator('ml-100k/u.data', 'ml-100k/u.item')
dg.gen_train_test(0.8, seed=42)

dg.write_csv('train.csv', dg.train, nb_states=[history_length], nb_actions=[ra_length])
dg.write_csv('test.csv', dg.test, nb_states=[history_length], nb_actions=[ra_length])

data = read_file('train.csv')

## Embeddings

if True:  # Generate embeddings?
    eg = EmbeddingsGenerator(dg.user_train,
                             pd.read_csv('ml-100k/u.data', sep='\t', names=['userId', 'itemId', 'rating', 'timestamp']))
    eg.train(nb_epochs=300)
    train_loss, train_accuracy = eg.test(dg.user_train)
    print('Train set: Loss=%.4f ; Accuracy=%.1f%%' % (train_loss, train_accuracy * 100))
    test_loss, test_accuracy = eg.test(dg.user_test)
    print('Test set: Loss=%.4f ; Accuracy=%.1f%%' % (test_loss, test_accuracy * 100))
    eg.save_embeddings('embeddings.csv')

files.upload()

embeddings = Embeddings(read_embeddings('embeddings.csv'))

state_space_size = embeddings.size() * history_length
action_space_size = embeddings.size() * ra_length

environment = Environment(data, embeddings, alpha, gamma, fixed_length)

tf.reset_default_graph()  # For multiple consecutive executions

sess = tf.Session()
# '1: Initialize actor network f_??^?? and critic network Q(s, a|??^??) with random weights'
actor = Actor(sess, state_space_size, action_space_size, batch_size, ra_length, history_length, embeddings.size(), tau,
              actor_lr)
critic = Critic(sess, state_space_size, action_space_size, history_length, embeddings.size(), tau, critic_lr)

train(sess, environment, actor, critic, embeddings, history_length, ra_length, buffer_size, batch_size, discount_factor,
      nb_episodes, filename_summary)

# Testing

dict_embeddings = {}
for i, item in enumerate(embeddings.get_embedding_vector()):
    str_item = str(item)
    assert (str_item not in dict_embeddings)
    dict_embeddings[str_item] = i


def state_to_items(state, actor, ra_length, embeddings, dict_embeddings, target=False):
    return [dict_embeddings[str(action)]
            for action in
            actor.get_recommendation_list(ra_length, np.array(state).reshape(1, -1), embeddings, target).reshape(
                ra_length, embeddings.size())]


def test_actor(actor, test_df, embeddings, dict_embeddings, ra_length, history_length, target=False, nb_rounds=1):
    ratings = []
    unknown = 0
    random_seen = []
    for _ in range(nb_rounds):
        for i in range(len(test_df)):
            history_sample = list(test_df[i].sample(history_length)['itemId'])
            recommendation = state_to_items(embeddings.embed(history_sample), actor, ra_length, embeddings,
                                            dict_embeddings, target)
            for item in recommendation:
                l = list(test_df[i].loc[test_df[i]['itemId'] == item]['rating'])
                assert (len(l) < 2)
                if len(l) == 0:
                    unknown += 1
                else:
                    ratings.append(l[0])
            for item in history_sample:
                random_seen.append(list(test_df[i].loc[test_df[i]['itemId'] == item]['rating'])[0])

    return ratings, unknown, random_seen


## Train set

#### Target = False

ratings, unknown, random_seen = test_actor(actor, dg.train, embeddings, dict_embeddings, ra_length, history_length,
                                           target=False, nb_rounds=10)
print('%0.1f%% unknown' % (100 * unknown / (len(ratings) + unknown)))

plt.subplot(1, 2, 1)
plt.hist(ratings)
plt.title('Predictions ; Mean = %.4f' % (np.mean(ratings)))
plt.subplot(1, 2, 2)
plt.hist(random_seen)
plt.title('Random ; Mean = %.4f' % (np.mean(random_seen)))
plt.show()

#### Target = True

ratings, unknown, random_seen = test_actor(actor, dg.train, embeddings, dict_embeddings, ra_length, history_length,
                                           target=True, nb_rounds=10)
print('%0.1f%% unknown' % (100 * unknown / (len(ratings) + unknown)))

plt.subplot(1, 2, 1)
plt.hist(ratings)
plt.title('Predictions ; Mean = %.4f' % (np.mean(ratings)))
plt.subplot(1, 2, 2)
plt.hist(random_seen)
plt.title('Random ; Mean = %.4f' % (np.mean(random_seen)))
plt.show()

## Test set

#### Target = False

ratings, unknown, random_seen = test_actor(actor, dg.test, embeddings, dict_embeddings, ra_length, history_length,
                                           target=False, nb_rounds=100)
print('%0.1f%% unknown' % (100 * unknown / (len(ratings) + unknown)))

plt.subplot(1, 2, 1)
plt.hist(ratings)
plt.title('Predictions ; Mean = %.4f' % (np.mean(ratings)))
plt.subplot(1, 2, 2)
plt.hist(random_seen)
plt.title('Random ; Mean = %.4f' % (np.mean(random_seen)))
plt.show()

#### Target = True

ratings, unknown, random_seen = test_actor(actor, dg.test, embeddings, dict_embeddings, ra_length, history_length,
                                           target=True, nb_rounds=100)
print('%0.1f%% unknown' % (100 * unknown / (len(ratings) + unknown)))

plt.subplot(1, 2, 1)
plt.hist(ratings)
plt.title('Predictions ; Mean = %.4f' % (np.mean(ratings)))
plt.subplot(1, 2, 2)
plt.hist(random_seen)
plt.title('Random ; Mean = %.4f' % (np.mean(random_seen)))
plt.show()
