# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:10:58 2019

@author: hplgit / jct
@source: https://github.com/hplgit/bumpy
"""
import numpy as np
import matplotlib.pylab as plt
import _pickle as cPickle
from generate_road_profiles import generate_bumpy_roads
from bumpy_model import bumpy_model


if __name__ == '__main__':
    
    # Input parameters as arguments here.
    # Assume units in M.K.S.
    generate_bumpy_roads(L=500, nroads=200, resolution=10)
    data = bumpy_model(m=60, b=25, k=60, v=5)
    
    # Root mean square values
    u_rms = [np.sqrt((1./len(u))*np.sum(u**2))
             for h, F, u in data[2:]]
    print ('u_rms:', u_rms)
    print ('Simulated for t in [0,%g]' % data[1][-1])

    # Save data list to file
    outfile = open('bumpy.res', 'wb')
    cPickle.dump(data, outfile)
    outfile.close()

    # =============== Post-processing begins ===============
    outfile = open('bumpy.res', 'rb')
    data = cPickle.load(outfile)
    outfile.close()

    # data = [x, t, [h, a, u], [h, a, u], ..., u_rms]
    x, t = data[0:2]
    

    # Plot u for second realization
#    realization = 1
#    u = data[2+realization][2][:]
#    plt.plot(t, u)
#    plt.title('Displacement')
#
#    
#    # Compute and plot velocity in second realization
#    dt = t[1] - t[0]
#    v = np.zeros_like(u)
#    v[1:-1] = (u[2:] - u[:-2])/(2*dt)
#    
#    # Estimate the starting and end velocity
#    v[0] = (u[1] - u[0])/dt
#    v[-1] = (u[-1] - u[-2])/dt
#    
#    plt.figure()
#    plt.plot(t, v)
#    plt.legend(['velocity'])
#    plt.xlabel('t')
#    plt.title('Velocity')
    
    
#    # Smooth the velocity (only internal points)
#    v[1:-1] = (v[2:] + 2*v[1:-1] + v[:-2])/4.0
#    plt.figure()
#    plt.plot(t, v)
#    plt.legend(['smoothed velocity'])
#    plt.xlabel('t')
#    plt.title('Velocity')
    
#    for realization in range(5):# len(data[2:])
#        h, F, u = data[2+realization]
#
#        plt.figure()
#        plt.subplot(3, 1, 1)
#        plt.plot(x, h, 'g-')
#        plt.legend(['h %d' % realization])
#        hmax = (abs(h.max()) + abs(h.min()))/2
#        plt.axis([x[0], x[-1], -hmax*5, hmax*5])
#        plt.xlabel('distance'); plt.ylabel('height')
#
#        plt.subplot(3, 1, 2)
#        plt.plot(t, F)
#        plt.legend(['F %d' % realization])
#        plt.xlabel('t'); plt.ylabel('force')
#
#        plt.subplot(3, 1, 3)
#        plt.plot(t, u, 'r-')
#        plt.legend(['u %d' % realization])
#        plt.xlabel('t'); plt.ylabel('displacement')
#
#    
#    plt.show()
    # =============== End of post-processing =============== 
 
    
    
    
    # =============== RNN-variations (ML) ===============
    # 嘗試在每筆資料中，取四點輸入，預測一點輸出(對於 label_output輸出而言，是downsampling)
    # Split trainning / test data
    train_cut = int(len(data[2:])*0.8)
    train_data = data[:train_cut]
    test_data = data[0:2]
    test_data.extend(data[train_cut:])
        
    # Source: Handson Tensorflow
    import tensorflow as tf
    
    # to make this output stable across runs
    def reset_graph(seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)


    
    def generate_batches(tv, n_steps, case_idx = 2):
        
        t_len = len(tv)
        indices = np.arange(0, t_len, n_steps) # np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
        ys_idx = indices[1:].reshape(-1, 1) # as local terminal
#        xs_idx = np.zeros((np.size(ys_idx, 0), n_steps)) # as local beginning
#        
#        for i in range(np.size(ys_idx, 0)):
#            xs_idx[i, :] = np.arange(indices[i], indices[i+1], 1)
        xs_idx = np.arange(0, indices[-1]).reshape(-1, 1)
            
        
        ys = train_data[case_idx][2][ys_idx] #.reshape(np.size(ys_idx, 0), 1, 1)
        xs = train_data[case_idx][1][xs_idx].reshape(np.size(ys_idx, 0), n_steps, 1)
        
        return xs, ys

    
    xs, ys = generate_batches(train_data[1], 4)
    
    reset_graph()

    n_steps = 4
    n_inputs = 1
    n_neurons = 10
    n_outputs = 1
    
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_outputs]) #[batch_size, n_steps = 1, n_outputs]
    
#    cell = tf.contrib.rnn.OutputProjectionWrapper(
#            tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
#            output_size=n_outputs)
    
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    
#    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
#    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
#    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    
    logits = tf.layers.dense(states, n_outputs)
    
    learning_rate = 0.01

    loss = tf.reduce_mean(tf.square(logits - y)) # MSE
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    
#    saver = tf.train.Saver()

    n_iterations = 500
    # batch_size = 50
    t_vect = train_data[1]
    
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            x_batch, y_batch = generate_batches(t_vect, n_steps, 2)
            sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
            if iteration % 10 == 0:
                mse = loss.eval(feed_dict={x: x_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)
        
#        saver.save(sess, "./my_time_series_model") 
        
    with tf.Session() as sess:
#        saver.restore(sess, "./my_time_series_model")   # not shown

        x_new, y_new = generate_batches(t_vect, n_steps, 3)
        y_pred = sess.run(logits, feed_dict={x: x_new})
