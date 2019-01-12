import numpy as np
import tensorflow as tf
import pandas as pd
from copy import deepcopy
import random
import math
import time


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            start = time.clock()
            direction = self.step()
            elapsed = (time.clock() - start)
            print("Time used:", elapsed)
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

#===========q-learning model===========#

#a graph for restoring the model
learned_graph = tf.Graph()


#depth of each layer
depth1 = 128  # conv layer1 depth
depth2 = 128  # conv layer2 depth
input_depth = 16  # input depth
hidden_units = 256  # fully connected hidden layer
output_units = 4  # output layer

# shape of weights
conv1_layer1_shape = [2, 1, input_depth, depth1]
conv1_layer2_shape = [2, 1, depth1, depth2]
conv2_layer1_shape = [1, 2, input_depth, depth1]
conv2_layer2_shape = [1, 2, depth1, depth2]

fc_layer1_w_shape = [3 * 4 * depth1 * 2 + 4 * 2 * depth2 * 2 + 3 * 3 * depth2 * 2, hidden_units]
fc_layer1_b_shape = [hidden_units]
fc_layer2_w_shape = [hidden_units, output_units]
fc_layer2_b_shape = [output_units]

parameters = dict()

path = './Final Weights'
parameters['conv1_layer1'] = np.array(pd.read_csv(path + '/conv1_layer1_weights.csv')['Weight']).reshape(conv1_layer1_shape)
parameters['conv1_layer2'] = np.array(pd.read_csv(path + '/conv1_layer2_weights.csv')['Weight']).reshape(conv1_layer2_shape)
parameters['conv2_layer1'] = np.array(pd.read_csv(path + '/conv2_layer1_weights.csv')['Weight']).reshape(conv2_layer1_shape)
parameters['conv2_layer2'] = np.array(pd.read_csv(path + '/conv2_layer2_weights.csv')['Weight']).reshape(conv2_layer2_shape)
parameters['fc_layer1_w'] = np.array(pd.read_csv(path + '/fc_layer1_weights.csv')['Weight']).reshape(fc_layer1_w_shape)
parameters['fc_layer1_b'] = np.array(pd.read_csv(path + '/fc_layer1_biases.csv')['Weight']).reshape(fc_layer1_b_shape)
parameters['fc_layer2_w'] = np.array(pd.read_csv(path + '/fc_layer2_weights.csv')['Weight']).reshape(fc_layer2_w_shape)
parameters['fc_layer2_b'] = np.array(pd.read_csv(path + '/fc_layer2_biases.csv')['Weight']).reshape(fc_layer2_b_shape)

with learned_graph.as_default():

    single_dataset = tf.placeholder(tf.float32, shape=(1, 4, 4, 16))  # input data :a 4*4*16 array

    # conv layer1 weights
    conv1_layer1_weights = tf.constant(parameters['conv1_layer1'], dtype=tf.float32)
    conv1_layer2_weights = tf.constant(parameters['conv1_layer2'], dtype=tf.float32)

    # conv layer2 weights
    conv2_layer1_weights = tf.constant(parameters['conv2_layer1'], dtype=tf.float32)
    conv2_layer2_weights = tf.constant(parameters['conv2_layer2'], dtype=tf.float32)

    # fully connected parameters
    fc_layer1_weights = tf.constant(parameters['fc_layer1_w'], dtype=tf.float32)
    fc_layer1_biases = tf.constant(parameters['fc_layer1_b'], dtype=tf.float32)
    fc_layer2_weights = tf.constant(parameters['fc_layer2_w'], dtype=tf.float32)
    fc_layer2_biases = tf.constant(parameters['fc_layer2_b'], dtype=tf.float32)

    # model
    def model(dataset):
        # conv layer1
        conv1 = tf.nn.conv2d(dataset, conv1_layer1_weights, [1, 1, 1, 1], padding='VALID')
        conv2 = tf.nn.conv2d(dataset, conv2_layer1_weights, [1, 1, 1, 1], padding='VALID')

        # layer1 relu activation
        relu1 = tf.nn.relu(conv1)
        relu2 = tf.nn.relu(conv2)

        # conv layer2
        conv11 = tf.nn.conv2d(relu1, conv1_layer2_weights, [1, 1, 1, 1], padding='VALID')
        conv12 = tf.nn.conv2d(relu1, conv2_layer2_weights, [1, 1, 1, 1], padding='VALID')
        conv21 = tf.nn.conv2d(relu2, conv1_layer2_weights, [1, 1, 1, 1], padding='VALID')
        conv22 = tf.nn.conv2d(relu2, conv2_layer2_weights, [1, 1, 1, 1], padding='VALID')

        # layer2 relu activation
        relu11 = tf.nn.relu(conv11)
        relu12 = tf.nn.relu(conv12)
        relu21 = tf.nn.relu(conv21)
        relu22 = tf.nn.relu(conv22)

        # get shapes of all activations
        shape1 = relu1.get_shape().as_list()
        shape2 = relu2.get_shape().as_list()

        shape11 = relu11.get_shape().as_list()
        shape12 = relu12.get_shape().as_list()
        shape21 = relu21.get_shape().as_list()
        shape22 = relu22.get_shape().as_list()

        # expansion
        hidden1 = tf.reshape(relu1, [shape1[0], shape1[1] * shape1[2] * shape1[3]])
        hidden2 = tf.reshape(relu2, [shape2[0], shape2[1] * shape2[2] * shape2[3]])

        hidden11 = tf.reshape(relu11, [shape11[0], shape11[1] * shape11[2] * shape11[3]])
        hidden12 = tf.reshape(relu12, [shape12[0], shape12[1] * shape12[2] * shape12[3]])
        hidden21 = tf.reshape(relu21, [shape21[0], shape21[1] * shape21[2] * shape21[3]])
        hidden22 = tf.reshape(relu22, [shape22[0], shape22[1] * shape22[2] * shape22[3]])

        # concatenation
        hidden = tf.concat([hidden1, hidden2, hidden11, hidden12, hidden21, hidden22], axis=1)

        # full connected layers
        hidden = tf.matmul(hidden, fc_layer1_weights) + fc_layer1_biases
        hidden = tf.nn.relu(hidden)

        # output layer
        output = tf.matmul(hidden, fc_layer2_weights) + fc_layer2_biases

        return output

    # for single example
    single_output = model(single_dataset)

#session for restore q-learning model
learned_sess = tf.Session(graph=learned_graph)

#change mat[4, 4] into power_mat[1, 4, 4, 16]
def change_the_map(X):
    power_mat = np.zeros(shape=(1, 4, 4, 16), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if X[i][j]==0:
                power_mat[0][i][j][0] = 1.0
            else:
                power = int(math.log(X[i][j], 2))
                power_mat[0][i][j][power] = 1.0
    return power_mat


#define my own agent
class Q_learningAgent(Agent):

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

        # learned_graph = tf.Graph()
        # learned_sess = tf.Session(graph=learned_graph)
        #
        # #depth of each layer
        # self.depth1 = 128  # conv layer1 depth
        # self.depth2 = 128  # conv layer2 depth
        # self.input_depth = 16  # input depth
        # self.hidden_units = 256  # fully connected hidden layer
        # self.output_units = 4  # output layer
        #
        # # shape of weights
        # self.conv1_layer1_shape = [2, 1, self.input_depth, self.depth1]
        # self.conv1_layer2_shape = [2, 1, self.depth1, self.depth2]
        # self.conv2_layer1_shape = [1, 2, self.input_depth, self.depth1]
        # self.conv2_layer2_shape = [1, 2, self.depth1, self.depth2]
        #
        # self.fc_layer1_w_shape = [3 * 4 * self.depth1 * 2 + 4 * 2 * self.depth2 * 2 + 3 * 3 * self.depth2 * 2, self.hidden_units]
        # self.fc_layer1_b_shape = [self.hidden_units]
        # self.fc_layer2_w_shape = [self.hidden_units, self.output_units]
        # self.fc_layer2_b_shape = [self.output_units]
        #
        # self.parameters = dict()
        #
        # self.path = r'./Final Weights'
        # self.parameters['conv1_layer1'] = np.array(pd.read_csv(self.path + r'/conv1_layer1_weights.csv')['Weight']).reshape(self.conv1_layer1_shape)
        # self.parameters['conv1_layer2'] = np.array(pd.read_csv(self.path + r'/conv1_layer2_weights.csv')['Weight']).reshape(self.conv1_layer2_shape)
        # self.parameters['conv2_layer1'] = np.array(pd.read_csv(self.path + r'/conv2_layer1_weights.csv')['Weight']).reshape(self.conv2_layer1_shape)
        # self.parameters['conv2_layer2'] = np.array(pd.read_csv(self.path + r'/conv2_layer2_weights.csv')['Weight']).reshape(self.conv2_layer2_shape)
        # self.parameters['fc_layer1_w'] = np.array(pd.read_csv(self.path + r'/fc_layer1_weights.csv')['Weight']).reshape(self.fc_layer1_w_shape)
        # self.parameters['fc_layer1_b'] = np.array(pd.read_csv(self.path + r'/fc_layer1_biases.csv')['Weight']).reshape(self.fc_layer1_b_shape)
        # self.parameters['fc_layer2_w'] = np.array(pd.read_csv(self.path + r'/fc_layer2_weights.csv')['Weight']).reshape(self.fc_layer2_w_shape)
        # self.parameters['fc_layer2_b'] = np.array(pd.read_csv(self.path + r'/fc_layer2_biases.csv')['Weight']).reshape(self.fc_layer2_b_shape)

    def step(self):
        output = learned_sess.run([single_output], feed_dict={single_dataset: change_the_map(self.game.board)})
        direction = np.argmax(output[0])
    
        # correction of parameter setting in training
        if direction == 1:    
           direction = 0
        elif direction == 3:
             direction = 1
        elif direction == 2:
             direction = 2
        elif direction == 0:
             direction = 3

        return direction
