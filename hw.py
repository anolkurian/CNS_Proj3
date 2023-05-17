# -*- coding: utf-8 -*-
""" CIS6261TML -- Homework 3 -- hw.py

# This file is the main homework file
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist

import scipy.stats as stats

# our neural network architectures
import nets
import attacks

## os / paths
def ensure_exists(dir_fp):
    if not os.path.exists(dir_fp):
        os.makedirs(dir_fp)

## parsing / string conversion to int / float
def is_int(s):
    try:
        z = int(s)
        return z
    except ValueError:
        return None


def is_number(s):
    try:
        z = int(s)
        return z
    except ValueError:
        try:
            z = float(s)
            return z
        except ValueError:
            return None


"""
## Save model to file
"""
def save_model(model, base_fp):
    # save the model: first the weights then the arch
    model.save_weights('{}-weights.h5'.format(base_fp))
    with open('{}-architecture.json'.format(base_fp), 'w') as f:
        f.write(model.to_json())


import hashlib

def memv_filehash(fp):
    hv = hashlib.sha256()
    buf = bytearray(512 * 1024)
    memv = memoryview(buf)
    with open(fp, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(memv), 0):
            hv.update(memv[:n])
    return hv.hexdigest()


"""
## Load model from file
"""        
def load_model(base_fp):
    # Model reconstruction from JSON file
    arch_json_fp = '{}-architecture.json'.format(base_fp)
    if not os.path.isfile(arch_json_fp):
        return None, None
        
    with open(arch_json_fp, 'r') as f:
        model = keras.models.model_from_json(f.read())

    wfp = '{}-weights.h5'.format(base_fp)

    # Load weights into the new model
    model.load_weights(wfp)
    
    hv = memv_filehash(wfp)
    
    print('Loaded model from file ({}) -- [{}].'.format(base_fp, hv[-17:-1].upper()))
    return model, hv
    
    
    

"""
## Load and preprocess the dataset
"""
def load_preprocess_mnist_data(train_size=50000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # MNIST has overall shape (60000, 28, 28) -- 60k images, each is 28x28 pixels
    print('Loaded mnist data; shape: {} [y: {}], test shape: {} [y: {}]'.format(x_train.shape, y_train.shape,
                                                                                      x_test.shape, y_test.shape))
    # Let's flatten the images for easier processing (labels don't change)
    flat_vector_size = 28 * 28
    x_train = x_train.reshape(x_train.shape[0], flat_vector_size)
    x_test = x_test.reshape(x_test.shape[0], flat_vector_size)
    
    assert x_train.shape[0] > train_size

    # Put the labels in "one-hot" encoding using keras' to_categorical()
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # let's split the training set further
    aux_idx = train_size

    x_aux = x_train[aux_idx:,:]
    y_aux = y_train[aux_idx:,:]

    x_temp = x_train[:aux_idx,:]
    y_temp = y_train[:aux_idx,:]

    x_train = x_temp
    y_train = y_temp

    return (x_train, y_train), (x_test, y_test), (x_aux, y_aux)


"""
## Plots a set of images (all m x m)
## input is  a square number of images, i.e., np.array with shape (z*z, dim_x, dim_y) for some integer z > 1
"""
def plot_images(im, dim_x=28, dim_y=28, one_row=False, out_fp='out.png', save=False, show=True, cmap='gray', fig_size=(14,14), titles=None, titles_fontsize=12):
    fig = plt.figure(figsize=fig_size)
    im = im.reshape((-1, dim_x, dim_y))

    num = im.shape[0]
    assert num <= 3 or np.sqrt(num)**2 == num or one_row, 'Number of images is too large or not a perfect square!'
    
    if titles is not None:
        assert num == len(titles)
    
    if num <= 3:
        for i in range(0, num):
            plt.subplot(1, num, 1 + i)
            plt.axis('off')
            if type(cmap) == list:
                assert len(cmap) == num
                plt.imshow(im[i], cmap=cmap[i]) # plot raw pixel data
            else:
                plt.imshow(im[i], cmap=cmap) # plot raw pixel data
            if titles is not None:
                plt.title(titles[i], fontsize=titles_fontsize)
    else:
        sq = int(np.sqrt(num))
        for i in range(0, num):
            if one_row:
                plt.subplot(1, num, 1 + i)
            else:
                plt.subplot(sq, sq, 1 + i)
            plt.axis('off')
            if type(cmap) == list:
                assert len(cmap) == num
                plt.imshow(im[i], cmap=cmap[i]) # plot raw pixel data
            else:
                plt.imshow(im[i], cmap=cmap) # plot raw pixel data
            if titles is not None:
                plt.title(titles[i], fontsize=titles_fontsize)

    if save:
        plt.savefig(out_fp)

    if show:
        plt.show()
    else:
        plt.close()



"""
## Extract 'sz' targets from in/out data
"""
def get_targets(x_in, y_in, x_out, y_out, sz=5000):
    
    x_temp = np.vstack((x_in, x_out))
    y_temp = np.vstack((y_in, y_out))

    inv = np.ones((x_in.shape[0],1))
    outv = np.zeros((x_out.shape[0],1))
    in_out_temp = np.vstack((inv, outv))

    assert x_temp.shape[0] == y_temp.shape[0]

    if sz > x_temp.shape[0]:
        sz = x_temp.shape[0]

    perm = np.random.permutation(x_temp.shape[0])
    perm = perm[0:sz]
    x_targets = x_temp[perm,:]
    y_targets = y_temp[perm,:]

    in_out_targets = in_out_temp[perm,:]

    return x_targets, y_targets, in_out_targets


## this is the main
def main():

    ######### Fill in your UFID here! ##############
    ufid = 56268544
    #for example: ufid = 12345678 # if your UFID is 1234-5678
    
    if ufid == 0 or ufid == 12345678:
        print('You must fill in your UFID first!')
        sys.exit(0)
       
        
    # set the seed for numpy and tensorflow based on the UFID
    np.random.seed(ufid)
    tf.random.set_seed(ufid)
    
    r = np.random.uniform() + tf.random.uniform((1,))[0]
    print('----- UFID: {} ; r: {:.6f}'.format(ufid, r))


    num_classes = 10 # mnist number of classes
    
    # figure out the problem number
    print(sys.argv)
    assert len(sys.argv) >= 3, 'Incorrect number of arguments!'
    p_split = sys.argv[1].split('problem')
    assert len(p_split) == 2 and p_split[0] == '', 'Invalid argument {}.'.format(sys.argv[1])
    problem_str = p_split[1]

    assert is_number(problem_str) is not None, 'Invalid argument {}.'.format(sys.argv[1])
    problem = float(problem_str)
    probno = int(problem)

    if probno < 0 or probno > 4:
        assert False, 'Problem {} is not a valid problem # for this assignment/homework!'.format(problem)

    ## change this line to override the verbosity behavior
    verb = True if probno == 0 else False

    # get arguments
    model_str = sys.argv[2]
    if model_str.startswith('simple'):
        simple_args = model_str.split(',')
        assert simple_args[0] == 'simple' and len(simple_args) == 3, '{} is not a valid network description string!'.format(model_str)
        hidden = is_int(simple_args[1])
        reg_const = is_number(simple_args[2])
        assert hidden is not None and hidden > 0 and reg_const is not None and reg_const >= 0.0, '{} is not a valid network description string!'.format(model_str)
        target_model_train_fn = lambda: nets.get_simple_classifier(num_hidden=hidden, l2_regularization_constant=reg_const,
                                                                   verbose=verb)
    elif model_str == 'deep':
        target_model_train_fn = lambda: nets.get_deeper_classifier(verbose=verb)
    else:
        assert False, '{} is not a valid network description string!'.format(model_str)

    # load the dataset
    train, test, aux = load_preprocess_mnist_data()
    x_train, y_train = train
    x_test, y_test = test
    x_aux, y_aux = aux
    
    x_aux = x_aux.astype(float)
    y_aux = y_aux.astype(float)
    
    # subsample training data
    tr_sz = 2000
    target_train_size = np.minimum(tr_sz, x_train.shape[0])
    x_train = x_train[0:target_train_size]
    y_train = y_train[0:target_train_size]
    
    # grab targets
    x_in = x_train
    y_in = y_train
    
    x_out = x_test[0:target_train_size]
    y_out = y_test[0:target_train_size]
    
    x_targets, y_targets, in_or_out_targets = get_targets(x_in, y_in, x_out, y_out, sz=tr_sz*2)


    dirpath = os.path.join(os.getcwd(), 'models')
    ensure_exists(dirpath)
    base_fp = '{}/{}'.format(dirpath, model_str)

    ########################################################
    ########### Model Training (Problem 1) #################
    ########################################################
    train_model = probno == 1 # train
    # train_model = 0 # train
    if train_model:
        assert len(sys.argv) == 4, 'Incorrect number of arguments!'
        model = target_model_train_fn()  # compile the target model
        
        num_epochs = is_int(sys.argv[3])

        assert num_epochs is not None and 0 < num_epochs <= 10000, '{} is not a valid size for the number of epochs to train the target model!'.format(sys.argv[3])

        # train the target model
        train_loss, train_accuracy, test_loss, test_accuracy = nets.train_model(model, x_train, y_train, x_test, y_test, num_epochs, verbose=verb)

        print('Trained target model on {} records. Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size,
                                                                                    100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))

        save_model(model, base_fp)
        
        sys.exit(0)
    else:
        model, _ = load_model(base_fp)
        target_model = model
        if model is None:
            print('Model files do not exist. Train the model first!'.format(base_fp))
            sys.exit(1)


    ## make sure the target model is not trainable (so we don't accidentally change the weights)
    model.trainable = False

    target_conf = 0.8    

    query_target_model = lambda x: target_model.predict(x, verbose=0)

    # y_pred = np.argmax(query_target_model(x_test), axis=1)
    # y_true = np.argmax(y_test, axis=1)

    # # get misclassified images
    # misclassified = np.where(y_pred != y_true)[0]
    # num_misclassified = len(misclassified)

    # print('Target model misclassified {}/{} images'.format(num_misclassified, len(x_test)))

    # if num_misclassified > 0:
    #     # plot the first 9 misclassified images
    #     plot_images(x_test[misclassified[:9]], titles=y_pred[misclassified[:9]], cmap='gray', fig_size=(8, 8), titles_fontsize=10)
    
    if probno == 2: ## problem 2

        assert len(sys.argv) == 6, 'Incorrect number of argument'
        
        # we need to know this to mimic the target model behavior
        num_epochs = is_int(sys.argv[3])

        num_shadow = is_int(sys.argv[4])
        attack_model_str = sys.argv[5]

        assert num_shadow is not None and 1 < num_shadow <= 200, '{} is not a valid number of shadow models!'.format(sys.argv[5])

        ## You can add new model types here
        if attack_model_str == 'LR':
            from sklearn.linear_model import LogisticRegression
            attack_model_fn = lambda : LogisticRegression(solver='lbfgs')
        elif attack_model_str == 'SVM':
            from sklearn.svm import LinearSVC
            attack_model_fn = LinearSVC
        elif attack_model_str == 'DT':
            from sklearn.tree import DecisionTreeClassifier
            attack_model_fn = DecisionTreeClassifier
        elif attack_model_str == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            attack_model_fn = RandomForestClassifier
        elif attack_model_str == 'NB':
            from sklearn.naive_bayes import GaussianNB
            attack_model_fn = GaussianNB
        elif attack_model_str == 'MLP':
            from sklearn.neural_network import MLPClassifier
            attack_model_fn = lambda : MLPClassifier(max_iter=400)
        else:
            assert False, '{} is not a valid attack model type!'.format(attack_model_str)

        create_model_fn = target_model_train_fn
        train_model_fn = lambda model, x, y: nets.train_model(model, x, y, None, None, num_epochs, verbose=False)

        attack_models = attacks.shokri_attack_models(x_aux, y_aux, target_train_size, create_model_fn, train_model_fn, num_shadow=num_shadow, attack_model_fn=attack_model_fn)


        in_or_out_pred = attacks.do_shokri_attack(attack_models, x_targets, y_targets, query_target_model)
        accuracy, advantage, _ = attacks.attack_performance(in_or_out_targets, in_or_out_pred)

        print('Shokri attack ({}) accuracy, advantage: {:.1f}%, {:.2f}'.format(attack_model_str, 100.0*accuracy, advantage))


    elif probno == 3:  ## problem 3

        assert len(sys.argv) == 3, 'Invalid extra argument'

        loss_fn = nets.compute_loss
        loss_train_vec = loss_fn(y_train, target_model.predict(x_train, verbose=0))
        loss_test_vec = loss_fn(y_test, target_model.predict(x_test, verbose=0))

        mean_train_loss = np.mean(loss_train_vec)
        std_train_loss = np.std(loss_train_vec)
        mean_test_loss = np.mean(loss_test_vec)
        std_test_loss = np.std(loss_test_vec)


        in_or_out_pred = attacks.do_loss_attack(x_targets, y_targets, query_target_model, loss_fn, mean_train_loss, std_train_loss, mean_test_loss, std_test_loss)
        accuracy, advantage, _ = attacks.attack_performance(in_or_out_targets, in_or_out_pred)

        print('Loss attack accuracy, advantage: {:.1f}%, {:.2f}'.format(100.0*accuracy, advantage))

        ## TODO ##
        ## Insert your code here to compute the best threshold (for loss_attack2)
        acc_list = []
        acc_posterior=[]
        threshold = [0.1,0.3,0.5,0.7,0.9]
        for t in threshold:
            in_or_out_pred = attacks.do_loss_attack2(x_targets, y_targets, query_target_model, loss_fn, mean_train_loss, std_train_loss, t)
            accuracy, advantage, _ = attacks.attack_performance(in_or_out_targets, in_or_out_pred)
            acc_list.append(accuracy)
        acc_list = [ round(elem, 2) for elem in acc_list ]
        idx = acc_list.index(max(acc_list))
        best_threshold_lossattack2 = threshold[idx]
        print("Best Threshold Loss Attack 2 :", best_threshold_lossattack2,max(acc_list))

        ## TODO ##
        plot_list = []
        for i in range(len(acc_list)):
            plot_list.append((threshold[i],acc_list[i]))
        labels,ys = zip(*plot_list)
        xs = np.arange(len(labels))
        width = 0.5
        plt.bar(xs,ys,width,align='center')
        plt.xticks(xs, labels)
        plt.yticks(ys)
        print("Loss2 accuracy:", acc_list)
        plt.savefig('Lossattack2_Threshold.png')

        # posterior attack
        for t in threshold:
            in_or_out_pred = attacks.do_posterior_attack(x_targets, y_targets, query_target_model, t )
            accuracy, advantage, _ = attacks.attack_performance(in_or_out_targets, in_or_out_pred)
            acc_posterior.append(accuracy)
        acc_posterior =  [ round(elem, 2) for elem in acc_posterior ]
        idx_post = acc_posterior.index(max(acc_posterior))
        best_threshold_posterior = threshold[idx_post]
        print('Posterior Attack Accuracy, Threshold: {:.1f}%, {:.2f}'.format(100.0*acc_posterior[idx_post], best_threshold_posterior))

        posterior_plot_list = []
        for i in range(len(acc_posterior)):
            posterior_plot_list.append((threshold[i],acc_posterior[i]))
        labels, ys = zip(*posterior_plot_list)
        xs = np.arange(len(labels))
        width = 0.5
        plt.bar(xs, ys, width, align='center')

        plt.xticks(xs, labels)
        plt.yticks(ys)
        print("Posterior accuracy:", acc_posterior)
        plt.savefig('Posterior_Threshold.png')

    elif probno == 4:  ## problem 4

        ## TODO ##
        ## Insert your code here [you can use plot_image()]
        num_epochs = is_int(sys.argv[3])
        num_shadow = is_int(sys.argv[4])
        print(sys.argv)
        print(num_shadow)
        from sklearn.neural_network import MLPClassifier
        attack_model_fn = lambda : MLPClassifier(max_iter=500)
        create_model_fn = target_model_train_fn
        train_model_fn = lambda model, x, y: nets.train_model(model, x, y, None, None, num_epochs, verbose=False)
        attack_models = attacks.shokri_attack_models(x_aux, y_aux, target_train_size, create_model_fn, train_model_fn,
                                                    num_shadow=num_shadow, attack_model_fn=attack_model_fn)
        in_or_out_pred_shokri = attacks.do_shokri_attack(attack_models, x_targets, y_targets, query_target_model)
        acc_shokri, advantage, _ = attacks.attack_performance(in_or_out_targets, in_or_out_pred_shokri)
        print('Shokri attack (MLP) accuracy, advantage: {:.1f}%, {:.2f}'.format( 100.0*acc_shokri, advantage))

        #Do Loss Attack
        loss_fn = nets.compute_loss
        loss_train_vec = loss_fn(y_train, target_model.predict(x_train))
        loss_test_vec = loss_fn(y_test, target_model.predict(x_test))

        mean_train_loss = np.mean(loss_train_vec)
        std_train_loss = np.std(loss_train_vec)
        mean_test_loss = np.mean(loss_test_vec)
        std_test_loss = np.std(loss_test_vec)

        #loss attack
        in_or_out_pred_loss = attacks.do_loss_attack(x_targets, y_targets, query_target_model, loss_fn, mean_train_loss, std_train_loss,
                                                mean_test_loss, std_test_loss)
        acc_loss, advantage, _ = attacks.attack_performance(in_or_out_targets, in_or_out_pred_loss)
        print('Loss attack accuracy, advantage: {:.1f}%, {:.2f}'.format(100.0*acc_loss, advantage))

        #Do Loss Attack 2
        in_or_out_pred_loss2 = attacks.do_loss_attack2(x_targets, y_targets, query_target_model, loss_fn, mean_train_loss, std_train_loss, 0.5)
        acc_loss2, advantage, _ = attacks.attack_performance(in_or_out_targets, in_or_out_pred_loss2)
        print('Loss attack2 accuracy, advantage: {:.1f}%, {:.2f}'.format(100.0*acc_loss2, advantage))

        #Do posterior attack
        in_or_out_pred_posterior = attacks.do_posterior_attack(x_targets, y_targets, query_target_model, 0.7 )
        acc_posterior, advantage, _ = attacks.attack_performance(in_or_out_targets, in_or_out_pred_posterior)
        print('posterior accuracy, advantage: {:.1f}%, {:.2f}'.format(100.0*acc_posterior, advantage))

        idx=[]
        idx = np.where(( in_or_out_targets == in_or_out_pred_shokri).all(axis=1))
        for i in range((len(idx))):
            plot_images(x_test[i])

if __name__ == '__main__':
    main()
