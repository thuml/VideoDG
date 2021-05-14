import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from module.Attmodules import *
from opts.opts import args
from opts.hyper import Hyperparams as hp





class APNRelationModuleMultiScale(torch.nn.Module):

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(APNRelationModuleMultiScale, self).__init__()
        self.subsample_num = 3
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]  

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(
                relations_scale)))  

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList()  
        self.transformer = AttModel(hp, 10000, 10000)

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_frames * self.img_feature_dim, self.num_class),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_frames * num_bottleneck, self.num_class)
        )


    def forward(self, input):
        act_all = input[:, self.relations_scales[0][0], :]
        act_all = self.transformer(act_all, act_all)
        act_block = act_all
        other = act_all
        adv_feature = act_all


        for scaleID in range(1, len(self.scales)):
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]),
                                                          self.subsample_scales[scaleID], replace=False)
            idx1, _, _ = idx_relations_randomsample
            act_relation1 = input[:, self.relations_scales[scaleID][idx1], :]

            act_relation1 = self.transformer(act_relation1, act_relation1)

            temp_1 = self.transformer(act_relation1, act_block)

            if scaleID == 1:
                other = temp_1
            else:
                other  =  other + temp_1
            act_all = act_all + temp_1

        other = other.view((-1, other.size(1) * other.size(2)))
        adv_result = self.final(other)

        act_feature = act_all.view((-1, act_all.size(1) * act_all.size(2)))


        act_all_result = self.final(act_feature)

        return act_all_result, act_feature, adv_result, other


    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))




def return_APN(relation_type, img_feature_dim, num_frames, num_class):
    if relation_type == 'APN':
        APNmodel = APNRelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    else:
        raise ValueError('Unknown' + relation_type)

    return APNmodel



class AttModel(nn.Module):
    def __init__(self, hp_, enc_voc, dec_voc):
        super(AttModel, self).__init__()
        self.hp = hp_

        self.enc_voc = enc_voc
        self.dec_voc = dec_voc

        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, multihead_attention(num_units=self.hp.hidden_units,
                                                                              num_heads=self.hp.num_heads,
                                                                              dropout_rate=self.hp.dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('dec_self_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

        self.logits_layer = nn.Linear(self.hp.hidden_units, self.dec_voc)
        self.label_smoothing = label_smoothing()

    def forward(self, x, y):
        self.enc = x
        for i in range(self.hp.num_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc)
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)
        self.dec = y
        for i in range(self.hp.num_blocks):
            self.dec = self.__getattr__('dec_self_attention_%d' % i)(self.dec, self.dec, self.dec)
            self.dec = self.__getattr__('dec_vanilla_attention_%d' % i)(self.dec, self.enc, self.enc)
            self.dec = self.__getattr__('dec_feed_forward_%d' % i)(self.dec)

        self.dec = self.dec_dropout(self.dec)

        return self.dec



