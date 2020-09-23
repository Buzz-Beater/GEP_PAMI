"""
Created on 12/9/18

@author: Baoxiong Jia

Description:

"""
import os
import numpy as np
import json

class Prob_Utils(object):

    @staticmethod
    def get_likelihood_sum(action_log_likelihood, object_log_likelihood=None, affordance_log_likelihood=None, affordance=True):
        '''
        Precompute the sum of log probabilities in interval [i, j]
        :param action_log_likelihood: action log likelihood
        :param object_log_likelihood: object log likelihood
        :param affordance_log_likelihood: affordance log likelihood
        :return:
                action_log_likelihood_sum: action_label_num x frames x frames
                object_log_likelihood_sum: object_bbox_num x object_label_num x frames x frames
                affordance_log_likelihood_sum: object_bbox_num x affordance_label_num x frames x frames
        '''
        action_log_likelihood_sum = np.zeros(
            action_log_likelihood.shape + (action_log_likelihood.shape[-1],))
        for a in range(action_log_likelihood.shape[0]):
            for i in range(action_log_likelihood.shape[1]):
                action_log_likelihood_sum[a, i, i] = action_log_likelihood[a, i]
        for a in range(action_log_likelihood.shape[0]):
            for i in range(action_log_likelihood.shape[1]):
                for j in range(i + 1, action_log_likelihood.shape[1]):
                    action_log_likelihood_sum[a, i, j] = action_log_likelihood_sum[a, i, j - 1] + \
                                                         action_log_likelihood[a, j]

        object_log_likelihood_sum = None
        affordance_log_likelihood_sum = None

        if affordance:
            object_log_likelihood_sum = np.zeros(object_log_likelihood.shape + (object_log_likelihood.shape[-1],))
            for b in range(object_log_likelihood.shape[0]):
                for o in range(object_log_likelihood.shape[1]):
                    for i in range(object_log_likelihood.shape[2]):
                        object_log_likelihood_sum[b, o, i, i] = object_log_likelihood[b, o, i]
            for b in range(object_log_likelihood.shape[0]):
                for o in range(object_log_likelihood.shape[1]):
                    for i in range(object_log_likelihood.shape[2]):
                        for j in range(i + 1, object_log_likelihood.shape[2]):
                            object_log_likelihood_sum[b, o, i, j] = object_log_likelihood_sum[b, o, i, j - 1] + \
                                                                    object_log_likelihood[b, o, j]

            affordance_log_likelihood_sum = np.zeros(
                affordance_log_likelihood.shape + (affordance_log_likelihood.shape[-1],))
            for b in range(affordance_log_likelihood.shape[0]):
                for a in range(affordance_log_likelihood.shape[1]):
                    for i in range(affordance_log_likelihood.shape[2]):
                        affordance_log_likelihood_sum[b, a, i, i] = affordance_log_likelihood[b, a, i]
            for b in range(affordance_log_likelihood.shape[0]):
                for a in range(affordance_log_likelihood.shape[1]):
                    for i in range(affordance_log_likelihood.shape[2]):
                        for j in range(i + 1, affordance_log_likelihood.shape[2]):
                            affordance_log_likelihood_sum[b, a, i, j] = affordance_log_likelihood_sum[b, a, i, j - 1] + \
                                                                        affordance_log_likelihood[b, a, j]

        return action_log_likelihood_sum, object_log_likelihood_sum, affordance_log_likelihood_sum

    @staticmethod
    def combine_cpt(action_log_cpt, object_log_cpt, affordance_log_cpt, affordance=True):
        '''
        Combine action prior, object prior and affordance prior using the log probability
        :param action_log_cpt: action log probability, indexed by (subactivity, action)
        :param object_log_cpt: object log probability, indexed by (subactivity, object)
        :param affordance_log_cpt: affordance log probability, indexed by (subactivity, affordance)
        :return: combined log probability, indexed by (subactivity, action, object, affordance)
        '''
        if affordance:
            combined_log_cpt = np.zeros((action_log_cpt.shape[0], action_log_cpt.shape[1],
                                     object_log_cpt.shape[1], affordance_log_cpt.shape[1]))
        else:
            combined_log_cpt = np.zeros((action_log_cpt.shape[0], action_log_cpt.shape[1]))
        for s in range(combined_log_cpt.shape[0]):
            for a in range(action_log_cpt.shape[1]):
                if affordance:
                    for o in range(object_log_cpt.shape[1]):
                        for u in range(affordance_log_cpt.shape[1]):
                            combined_log_cpt[s, a, o, u] = action_log_cpt[s, a] + object_log_cpt[s, o] + affordance_log_cpt[s, u]
                else:
                    combined_log_cpt[s, a] = action_log_cpt[s, a]
        return combined_log_cpt