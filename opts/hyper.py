# -*- coding: utf-8 -*-
# /usr/bin/python2



class Hyperparams:
  
    batch_size = 32 
    lr = 0.0001  
    logdir = 'logdir'  

    model_dir = './models/'  

    maxlen = 2048  
    min_cnt = 20  
    hidden_units = 1 * 256  
    num_blocks = 1  
    num_epochs = 20
    num_heads = 4  
    dropout_rate = 0.3  
    sinusoid = True  
    eval_epoch = 20  
    preload = None 



