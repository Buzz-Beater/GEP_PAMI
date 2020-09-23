"""
Created on 11/27/18

@author: Baoxiong Jia

Description:

"""
import sys
sys.path.append('/media/hdd/home/baoxiong/Projects/TPAMI2019/src')

import os
import time
import json
import copy
import pickle

import scipy.stats
import numpy as np

# import datasets.VCLA_GAZE.metadata as metadata
# import datasets.VCLA_GAZE.vcla_gaze_config as config
import datasets.CAD.cad_config as config
import datasets.CAD.metadata as metadata
from models.parser import grammarutils
from models import parsegraph as parsegraph
from experiments.STAOG.prob_utils import Prob_Utils
import utils.logutils as utils


def load_prior(paths, affordance=True):
    '''
    Load prior distribution for Bayes inference, including action, object, affordance and duration prior
    :param paths: config.Paths() object
    :return: action_log_cpt, object_log_cpt, affordance_log_cpt, duration_prior, combined_log_cpt
    '''
    with open(os.path.join(os.path.join(paths.prior_root, 'action_cpt.json'))) as f:
        action_log_cpt = np.log(json.load(f))
    with open(os.path.join(paths.prior_root, 'duration_prior.json')) as f:
        duration_prior = json.load(f)
    object_log_cpt = None
    affordance_log_cpt = None
    if affordance:
        with open(os.path.join(os.path.join(paths.prior_root, 'object_cpt.json'))) as f:
            object_log_cpt = np.log(json.load(f))
        with open(os.path.join(os.path.join(paths.prior_root, 'affordance_cpt.json'))) as f:
            affordance_log_cpt = np.log(json.load(f))

    combined_log_cpt = Prob_Utils.combine_cpt(action_log_cpt, object_log_cpt, affordance_log_cpt)
    return action_log_cpt, object_log_cpt, affordance_log_cpt, duration_prior, combined_log_cpt

def get_duration_distribution_params(duration_prior):
    '''
    Extract duration information from duration prior files
    :param duration_prior: the json file loaded
    :return: duration parameters (Gaussian) for each subactivity category
    '''
    duration_distribution_params = [None] * len(duration_prior.keys())
    for subactivity, params in duration_prior.items():
        duration_distribution_params[metadata.subactivity_index[subactivity]] = params
    return duration_distribution_params

def get_intermediate_results(paths, tpg, verbose=False):
    '''

    :param paths:
    :param tpg:
    :param verbose:
    :return:
    '''
    best_prob = 0.9
    small_prob = 1 - best_prob
    log_best_prob = np.log(best_prob)
    start_frame = tpg.start_frame
    frames = tpg.length + 1
    object_num = len(tpg.terminals[0].objects)
    #
    # action_log_likelihood = np.load(os.path.join(paths.inter_root, 'likelihood', 'activity',
    #                                                                 tpg.activity + '$' + tpg.id + '.npy')).T
    action_log_likelihood = np.load(
        os.path.join(paths.tmp_root, 'intermediate', 'action', tpg.subject.lower(), tpg.id + '.npy')).T

    action_log_likelihood = (action_log_likelihood + small_prob) / (1 + small_prob * len(metadata.actions))
    action_log_likelihood = np.log(action_log_likelihood)
    # affordance_log_likelihood = np.load(os.path.join(paths.inter_root, 'likelihood', 'affordance', tpg.activity + '$' + tpg.id + '.npy'))
    # if len(affordance_log_likelihood.shape) != 3:
    #     affordance_log_likelihood = np.expand_dims(affordance_log_likelihood, axis=1)
    affordance_log_likelihood = np.load(
        os.path.join(paths.tmp_root, 'intermediate', 'affordance', tpg.subject.lower(), tpg.id + '.npy'))
    # affordance_log_likelihood = np.transpose(affordance_log_likelihood, (1, 2, 0))
    affordance_log_likelihood = np.transpose(affordance_log_likelihood, (0, 2, 1))
    affordance_log_likelihood = (affordance_log_likelihood + small_prob) / (1 + small_prob * len(metadata.affordances))
    affordance_log_likelihood = np.log(affordance_log_likelihood)

    gt_action_log_likelihood = np.ones((len(metadata.actions), frames)) * np.log(
        (1 - best_prob) / (len(metadata.actions) - 1))
    gt_object_log_likelihood = np.ones((object_num, len(metadata.objects), frames)) * np.log(
        (1 - best_prob) / (len(metadata.objects) - 1))
    gt_affordance_log_likelihood = np.ones((object_num, len(metadata.affordances), frames)) * np.log(
        (1 - best_prob) / (len(metadata.affordances) - 1))

    # ========== Ground truth likelihoods
    for spg in tpg.terminals:
        gt_action_log_likelihood[metadata.action_index[spg.subactivity],
        spg.start_frame - start_frame: spg.end_frame + 1 - start_frame] = log_best_prob

        for i in range(len(spg.objects)):
            gt_object_log_likelihood[i, metadata.object_index[spg.objects[i]],
            spg.start_frame - start_frame: spg.end_frame + 1 - start_frame] = log_best_prob

        for i in range(len(spg.affordance)):
            gt_affordance_log_likelihood[i, metadata.affordance_index[spg.affordance[i]],
            spg.start_frame - start_frame:spg.end_frame + 1 - start_frame] = log_best_prob

    # ========== Fill in the missing frames
    action_log_likelihood = np.hstack(
        (action_log_likelihood, gt_action_log_likelihood[:, action_log_likelihood.shape[1]:]))
    affordance_log_likelihood = np.concatenate(
        (affordance_log_likelihood, gt_affordance_log_likelihood[:, :, affordance_log_likelihood.shape[2]:]), axis=2)

    with open(os.path.join(paths.prior_root, 'object_affordance_cpt.json')) as f:
        object_affordance_cpt = np.log(np.array(json.load(f)))
    for io in range(affordance_log_likelihood.shape[0]):
        o = np.argmax(gt_object_log_likelihood[io, :, 0])
        object_prior = np.tile(object_affordance_cpt[o, :], (frames, 1))
        affordance_log_likelihood[io, :, :] = affordance_log_likelihood[io, :, :] + object_prior.T

    return Prob_Utils.get_likelihood_sum(action_log_likelihood, gt_object_log_likelihood, affordance_log_likelihood)

def get_gt_intermediate_results(paths, tpg):
    '''

    :param paths:
    :param tpg:
    :return:
    '''
    best_prob = 0.9
    log_best_prob = np.log(best_prob)
    start_frame = tpg.start_frame
    frames = tpg.length
    object_num = len(tpg.terminals[0].objects)

    action_log_likelihood = np.ones((len(metadata.actions), frames)) * np.log(
        (1 - best_prob) / (len(metadata.actions) - 1))
    object_log_likelihood = np.ones((object_num, len(metadata.objects), frames)) * np.log(
        (1 - best_prob) / (len(metadata.objects) - 1))
    affordance_log_likelihood = np.ones((object_num, len(metadata.affordances), frames)) * np.log(
        (1 - best_prob) / (len(metadata.affordances) - 1))

    for spg in tpg.terminals:
        action_log_likelihood[metadata.action_index[spg.subactivity],
        spg.start_frame - start_frame: spg.end_frame + 1 - start_frame] = log_best_prob

        for i in range(len(spg.objects)):
            object_log_likelihood[i, metadata.object_index[spg.objects[i]],
            spg.start_frame - start_frame: spg.end_frame + 1 - start_frame] = log_best_prob

        for i in range(len(spg.affordance)):
            affordance_log_likelihood[i, metadata.affordance_index[spg.affordance[i]],
            spg.start_frame - start_frame:spg.end_frame + 1 - start_frame] = log_best_prob

    return Prob_Utils.get_likelihood_sum(action_log_likelihood, object_log_likelihood, affordance_log_likelihood)

def get_perturbed_intermediate_results(paths, tpg):
    '''

    :param paths:
    :param tpg:
    :return:
        action_log_likelihood: action_label_num x frames
        object_log_likelihood: object_bbox_num x object_label_num x frames
        affordance_log_likelihood: object_bbox_num x affordance_label_num x frames
    '''

    best_prob = 0.9
    perturb_prob = 0.1

    log_best_prob = np.log(best_prob)
    start_frame = tpg.start_frame
    frames = tpg.length
    object_num = len(tpg.terminals[0].objects)

    action_log_likelihood = np.ones((len(metadata.actions), frames)) * np.log((1 - best_prob) / (len(metadata.actions) - 1))
    object_log_likelihood = np.ones((object_num, len(metadata.objects), frames)) * np.log((1 - best_prob) / (len(metadata.objects) - 1))
    affordance_log_likelihood = np.ones((object_num, len(metadata.affordances), frames)) * np.log((1 - best_prob) / (len(metadata.affordances) - 1))

    for spg in tpg.terminals:
        for f in range(spg.start_frame-start_frame, spg.end_frame+1-start_frame):
            if np.random.rand() < perturb_prob:
                s = np.random.choice(range(len(metadata.actions)))
                action_log_likelihood[s, f] = log_best_prob
            else:
                action_log_likelihood[metadata.action_index[spg.subactivity], f] = log_best_prob

            for i in range(len(spg.objects)):
                object_log_likelihood[i, metadata.object_index[spg.objects[i]], f] = log_best_prob

            for i in range(len(spg.affordance)):
                if np.random.rand() < perturb_prob:
                    a = np.random.choice(range(len(metadata.affordances)))
                    affordance_log_likelihood[i, a, f] = log_best_prob
                else:
                    affordance_log_likelihood[i, metadata.affordance_index[spg.affordance[i]], f] = log_best_prob

    return Prob_Utils.get_likelihood_sum(action_log_likelihood, object_log_likelihood, affordance_log_likelihood)

def get_ground_truth_label(tpg):
    frames = tpg.length
    gt_subactivity = list()
    gt_objects = list()
    gt_affordance = list()
    for spg in tpg.terminals:
        gt_subactivity.extend([metadata.subactivity_index[spg.subactivity]]*(spg.end_frame-spg.start_frame+1))
        gt_objects.extend([[metadata.object_index[o] for o in spg.objects] for _ in range(spg.end_frame-spg.start_frame+1)])
        gt_affordance.extend([[metadata.affordance_index[u] for u in spg.affordance] for _ in range(spg.end_frame-spg.start_frame+1)])
    assert frames == len(gt_subactivity)
    return np.array(gt_subactivity), np.array(gt_objects).T, np.array(gt_affordance).T


def get_label(tpg, obj_num):
    start_frame = tpg.start_frame
    frames = tpg.length
    subactivities = np.empty(frames, dtype=int)
    actions = np.empty(frames, dtype=int)
    objects = np.empty((obj_num, frames), dtype=int)
    affordance = np.empty((obj_num, frames), dtype=int)
    for spg in tpg.terminals:
        for frame in range(spg.start_frame-1, spg.end_frame):
            subactivities[frame-start_frame] = spg.subactivity
            actions[frame-start_frame] = spg.action
            objects[:, frame-start_frame] = spg.objects
            affordance[:, frame-start_frame] = spg.affordance

    return subactivities, actions, objects, affordance

def dp_segmentation_s(priors, likelihoods):
    def segment_max_prob(b, f):
        #TODO Adjust DP iterating formula
        """
            a: action label index
            o: object label index
            u: affordance label index
            s: sub-activity label index
        :param b:
        :param f:
        :return:
        """
        # Default subactivity: null
        if f - b < 0:
            return 0, 0, -np.inf
        s = metadata.subactivity_index['null']
        ps = 0

        a = np.argmax(action_log_likelihood_sum[:, b, f])
        pa = action_log_likelihood_sum[a, b, f]

        # TODO: test infer of s
        s = np.argmax(action_log_cpt[:valid_s_count, a])
        mu, std = duration_distribution_params[s]
        # # TODO: penalize small time spans
        # if f-b < 0:
        #     ps = combined_log_cpt[s, a] + small_log_prob
        # else:
        #     #ps = combined_log_cpt[s, a] + small_log_prob
        #     ps = combined_log_cpt[s, a] + scipy.stats.norm.logpdf(f-b, mu, std)

        # Modify log_prob = pa #+ ps
        log_prob = pa
        return a, s, log_prob

    small_log_prob = np.log(0.000001)
    action_log_cpt, object_log_cpt, affordance_log_cpt, duration_prior, combined_log_cpt = priors
    duration_distribution_params = get_duration_distribution_params(duration_prior)
    action_log_likelihood_sum, object_log_likelihood_sum, affordance_log_likelihood_sum = likelihoods

    #TODO change to Watch_n_Patch num
    valid_s_count = len(metadata.subactivities) - 1
    frames = action_log_likelihood_sum.shape[-1]
    log_probs = np.empty(frames)
    trace_a = np.empty(frames, dtype=int)
    trace_s = np.empty(frames, dtype=int)
    trace_begin = np.empty(frames, dtype=int)

    # Segment the sequence by dynamic programming
    begin = 0
    for end in range(frames):
        a, s, log_prob = segment_max_prob(begin, end)
        trace_a[end] = a
        trace_s[end] = s
        log_probs[end] = log_prob
        trace_begin[end] = begin

    for end in range(1, frames):
        for begin in range(end):
            a, s, log_prob = segment_max_prob(begin, end)
            if log_probs[begin-1] + log_prob > log_probs[end]:
                trace_a[end] = a
                trace_s[end] = s
                log_probs[end] = log_prob
                trace_begin[end] = begin

    return trace_begin, trace_a, trace_s

def dp_segmentation(priors, likelihoods):
    def segment_max_prob(b, f):
        """
            a: action label index
            o: object label index
            u: affordance label index
            s: sub-activity label index
        :param b:
        :param f:
        :return:
        """
        # Default subactivity: null
        if f - b < 0:
            return 0, 0, 0, 0, -np.inf
        s = metadata.subactivity_index['null']
        ps = 0

        a = np.argmax(action_log_likelihood_sum[:, b, f])
        pa = action_log_likelihood_sum[a, b, f]
        # TODO: test infer of s
        s = np.argmax(action_log_cpt[:valid_s_count, a])

        o = np.empty(object_num, dtype=int)
        po = 0
        for io in range(object_num):
            o[io] = np.argmax(object_log_likelihood_sum[io, :, b, f])
            po += object_log_likelihood_sum[io, o[io], b, f]

        u = np.empty(object_num, dtype=int)
        pu = 0
        for iu in range(object_num):
            u[iu] = np.argmax(affordance_log_likelihood_sum[iu, :, b, f])
            pu += affordance_log_likelihood_sum[iu, u[iu], b, f]

            # TODO: more complicated labeling of s: no object or multiple objects
            if u[iu] != metadata.affordance_index['stationary']:
                s = np.argmax(combined_log_cpt[:valid_s_count, a, o[iu], u[iu]])
                mu, std = duration_distribution_params[s]
                # TODO: penalize small time spans
                if f-b < 0:
                    ps = combined_log_cpt[s, a, o[iu], u[iu]] + small_log_prob
                else:
                    ps = combined_log_cpt[s, a, o[iu], u[iu]] + scipy.stats.norm.logpdf(f-b, mu, std)

        log_prob = pa + po + pu + ps
        return a, o, u, s, log_prob

    small_log_prob = np.log(0.000001)
    action_log_cpt, object_log_cpt, affordance_log_cpt, duration_prior, combined_log_cpt = priors
    duration_distribution_params = get_duration_distribution_params(duration_prior)
    action_log_likelihood_sum, object_log_likelihood_sum, affordance_log_likelihood_sum = likelihoods
    object_num = object_log_likelihood_sum.shape[0]
    valid_s_count = len(metadata.subactivities) - 1

    frames = action_log_likelihood_sum.shape[-1]
    log_probs = np.empty(frames)
    trace_a = np.empty(frames, dtype=int)
    trace_o = np.empty((object_num, frames), dtype=int)
    trace_u = np.empty((object_num, frames), dtype=int)
    trace_s = np.empty(frames, dtype=int)
    trace_begin = np.empty(frames, dtype=int)

    # Segment the sequence by dynamic programming
    begin = 0
    for end in range(frames):
        a, o, u, s, log_prob = segment_max_prob(begin, end)
        trace_a[end] = a
        trace_o[:, end] = o
        trace_u[:, end] = u
        trace_s[end] = s
        log_probs[end] = log_prob
        trace_begin[end] = begin

    for end in range(1, frames):
        for begin in range(end):
            a, o, u, s, log_prob = segment_max_prob(begin, end)
            if log_probs[begin-1] + log_prob > log_probs[end]:
                trace_a[end] = a
                trace_o[:, end] = o
                trace_u[:, end] = u
                trace_s[end] = s
                log_probs[end] = log_prob
                trace_begin[end] = begin

    return trace_begin, trace_a, trace_o, trace_u, trace_s

def trace_label(trace_begin, trace, last_frame):
    '''

    :param trace_begin:
    :param trace:
    :param last_frame:
    :return:
    '''
    # Back trace labels
    labels = np.empty(last_frame, dtype=int)
    end_frame = last_frame
    while end_frame != 0:
        end_frame -= 1
        begin_frame = trace_begin[end_frame]
        for frame in range(begin_frame, end_frame + 1):
            labels[frame] = trace[end_frame]
        end_frame = begin_frame
    return labels


def generate_parse_graph(trace_begin, trace_a, trace_o, trace_u, trace_s, start_frame, end_frame):
    '''

    :param trace_begin:
    :param trace_a:
    :param trace_o:
    :param trace_u:
    :param trace_s:
    :param start_frame:
    :param end_frame:
    :return:
    '''
    subactivity_lables = trace_label(trace_begin, trace_s, end_frame)
    action_labels = trace_label(trace_begin, trace_a, end_frame)

    object_labels = np.empty((trace_o.shape[0], end_frame), dtype=int)
    affordance_labels = np.empty((trace_o.shape[0], end_frame), dtype=int)
    for o in range(trace_o.shape[0]):
        object_labels[o, :] = trace_label(trace_begin, trace_o[o, :], end_frame)
        affordance_labels[o, :] = trace_label(trace_begin, trace_u[o, :], end_frame)

    tpg = parsegraph.TParseGraph()
    seg_start = 0
    for frame in range(end_frame):
        if subactivity_lables[frame] != subactivity_lables[seg_start]:
            spg = parsegraph.SParseGraph(start_frame + seg_start, start_frame + frame - 1, subactivity_lables[seg_start], action_labels[seg_start], object_labels[:, seg_start], affordance_labels[:, seg_start])
            tpg.append_terminal(spg)
            seg_start = frame
    spg = parsegraph.SParseGraph(start_frame + seg_start, start_frame + end_frame - 1, subactivity_lables[seg_start], action_labels[seg_start], object_labels[:, seg_start], affordance_labels[:, seg_start])
    tpg.append_terminal(spg)

    return tpg

def tpg_to_tokens(tpg, end_frame):
    tokens = list()
    for spg in tpg.terminals:
        if spg.end_frame > end_frame:
            break
        tokens.append(metadata.subactivities[spg.subactivity])
    return tokens

def compute_pg_posterior(temperature, tpg, grammar, language, priors, likelihoods, end_frame):
    action_log_cpt, object_log_cpt, affordance_log_cpt, duration_prior, combined_log_cpt = priors
    duration_distribution_params = get_duration_distribution_params(duration_prior)
    action_log_likelihood_sum, object_log_likelihood_sum, affordance_log_likelihood_sum = likelihoods

    # Compute grammar prior
    tokens = tpg_to_tokens(tpg, end_frame)
    prior = grammarutils.compute_sentence_probability(grammar, language, tokens)
    if prior == 0:
        # warnings.warn('Prior is zero.')
        return 0
    log_prior = np.log(prior)

    log_likelihood = 0
    start_frame = tpg.terminals[0].start_frame
    for spg in tpg.terminals:
        if spg.end_frame > end_frame:
            break

        mu, std = duration_distribution_params[spg.subactivity]
        log_prior += scipy.stats.norm.logpdf(spg.end_frame-spg.start_frame, mu, std)
        log_prior += action_log_cpt[spg.subactivity, spg.action]
        log_likelihood += action_log_likelihood_sum[spg.action, spg.start_frame-start_frame, spg.end_frame-start_frame]

        for o in range(spg.objects.shape[0]):
            log_prior += object_log_cpt[spg.subactivity, spg.objects[o]]
            log_prior += affordance_log_cpt[spg.subactivity, spg.affordance[o]]
            log_likelihood += object_log_likelihood_sum[o, spg.objects[o], spg.start_frame-start_frame, spg.end_frame-start_frame]
            log_likelihood += affordance_log_likelihood_sum[o, spg.affordance[o], spg.start_frame-start_frame, spg.end_frame-start_frame]

    # warnings.filterwarnings('error')
    # try:
    #     foo = np.exp((log_prior+log_likelihood)/temperature)
    # except Warning:
    #     print (log_prior+log_likelihood), temperature, ((log_prior+log_likelihood)/temperature)
    return np.exp((log_prior+log_likelihood)/temperature)
    # return (log_prior+log_likelihood)/temperature

def sample_subactivity_length(params, min_length, trial_limit=10):
    '''

    :param params:
    :param min_length:
    :param trial_limit:
    :return:
    '''
    mu, std = params
    sample_length = int(np.random.normal(mu, std))
    trials = 0
    while sample_length < min_length:
        sample_length = int(np.random.normal(mu, std))
        trials += 1
        if trials >= trial_limit:
            return min_length + 5
    return sample_length

def gibbs_sampling(tpg, grammar_dict, languages, priors, likelihoods):
    max_posterior = -np.inf
    best_tpg = None
    for activity in metadata.activities:
        # if activity != 'stacking_objects':
        #     continue
        if activity != tpg.activity:
            continue
        grammar = grammar_dict[activity]
        language = languages[activity]
        tpg_copy = copy.deepcopy(tpg)
        tpg_copy.activity = activity

        modifed = True
        sample_iteration = 1
        temperature = 1.0
        while modifed:
            modifed = False
            # for spg in tpg_copy.terminals:
            #     # Sample sub-activity label
            #     current_s = spg.subactivity
            #     # Eliminate the 'prior', which is a invalid subactivity label
            #     posteriors = np.empty(len(metadata.subactivities)-1)
            #     for s in range(len(metadata.subactivities)-1):
            #         spg.subactivity = s
            #         posteriors[s] = compute_pg_posterior(temperature, tpg_copy, grammar, language, priors, likelihoods, spg.end_frame)
            #
            #     if np.sum(posteriors) == 0:
            #             # warnings.warn('Posteriors are 0 for all labels')
            #         spg.subactivity = current_s
            #         continue
            #
            #     posteriors = posteriors/np.sum(posteriors)
            #     sampled_s = np.random.choice(posteriors.shape[0], 1, p=posteriors)[0]
            #     spg.subactivity = sampled_s
            #     if sampled_s != current_s:
            #         modifed = True
            #
            #     # Sample affordance label
            #     # TODO
            #     for io in range(len(spg.objects)):
            #         current_o = spg.objects[io]
            #         # Eliminate the 'prior', which is a invalid subactivity label
            #         posteriors = np.empty(len(metadata.objects))
            #         for o in range(len(metadata.objects)):
            #             spg.objects[io] = o
            #             posteriors[o] = compute_pg_posterior(temperature, tpg_copy, grammar, language, priors, likelihoods, spg.end_frame)
            #
            #         if np.sum(posteriors) == 0:
            #             # warnings.warn('Posteriors are 0 for all labels')
            #             spg.objects[io] = current_o
            #             continue
            #
            #         posteriors = posteriors/np.sum(posteriors)
            #         sampled_o = np.random.choice(posteriors.shape[0], 1, p=posteriors)[0]
            #         spg.objects[io] = sampled_o
            #         if sampled_o != current_o:
            #             modifed = True
            #
            #     temperature *= 0.9

            final_posterior = compute_pg_posterior(temperature, tpg_copy, grammar, language, priors, likelihoods, np.inf)
            if final_posterior > max_posterior:
                max_posterior = final_posterior
                best_tpg = copy.deepcopy(tpg_copy)
                # print final_posterior, activity_label, best_tpg.activity

    # print best_tpg.activity
    return best_tpg

def predict(grammar_dict, languages, tpg, frame, duration, priors, likelihoods):
    action_log_cpt, object_log_cpt, affordance_log_cpt, duration_prior, combined_log_cpt = priors
    duration_distribution_params = get_duration_distribution_params(duration_prior)
    action_log_likelihood_sum, object_log_likelihood_sum, affordance_log_likelihood_sum = likelihoods
    predicted_tpg = copy.deepcopy(tpg)

    s = predicted_tpg.terminals[-1].subactivity
    sample_length = sample_subactivity_length(duration_distribution_params[s], predicted_tpg.terminals[-1].end_frame - predicted_tpg.terminals[-1].start_frame + 1)
    predicted_tpg.terminals[-1].end_frame = predicted_tpg.terminals[-1].start_frame + sample_length
    while predicted_tpg.terminals[-1].end_frame <= frame + duration - 1:
        # Sample a new spg
        d, matched_tokens = grammarutils.find_closest_tokens(languages[predicted_tpg.activity], tpg_to_tokens(predicted_tpg, np.inf))
        symbols, probs = grammarutils.predict_next_symbols(grammar_dict[predicted_tpg.activity], matched_tokens)
        if not symbols:
            break
        probs = np.array(probs)/np.sum(probs)
        sampled_symbol = symbols[np.random.choice(probs.shape[0], 1, p=probs)[0]]
        sampled_s = metadata.subactivity_index[sampled_symbol]
        sample_length = sample_subactivity_length(duration_distribution_params[sampled_s], 1)

        # Sample the action, affordance labels
        pa = action_log_cpt[sampled_s, :]*action_log_likelihood_sum[:, frame, frame]
        a = np.argmax(pa)
        o = predicted_tpg.terminals[-1].objects
        u = [None] * len(o)
        for io in range(len(o)):
            # TODO
            # pu = affordance_log_cpt[sampled_s, :]*affordance_log_likelihood_sum[io, :, frame, frame]

            likelihood_frame = min(affordance_log_likelihood_sum.shape[3]-1, frame + duration - 1)
            pu = affordance_log_cpt[sampled_s, :]*affordance_log_likelihood_sum[io, :, likelihood_frame, likelihood_frame]
            u[io] = np.argmax(pu)

            # u[io] = tpg.terminals[-1].affordance[io]  # TODO

        spg = parsegraph.SParseGraph(predicted_tpg.terminals[-1].end_frame + 1, predicted_tpg.terminals[-1].end_frame + sample_length, sampled_s, a, o, u)
        predicted_tpg.append_terminal(spg)

    return predicted_tpg

def get_next_subactivity_label(gt_tpg, predicted_tpg, seg_gt_s, seg_pred_s, end_frame):
    for i_spg, spg in enumerate(gt_tpg.terminals):
        if spg.start_frame <= end_frame <= spg.end_frame:
            if i_spg+1 < len(gt_tpg.terminals) and i_spg+1 < len(predicted_tpg.terminals):
                seg_gt_s.append(metadata.subactivity_index[gt_tpg.terminals[i_spg+1].subactivity])
                seg_pred_s.append(predicted_tpg.terminals[i_spg+1].subactivity)


def get_next_affordance_label(gt_tpg, predicted_tpg, seg_gt_u, seg_pred_u, obj_num, end_frame):
    for i_spg, spg in enumerate(gt_tpg.terminals):
        if spg.start_frame <= end_frame <= spg.end_frame:
            if i_spg+1 < len(gt_tpg.terminals) and i_spg+1 < len(predicted_tpg.terminals):
                for io in range(obj_num):
                    seg_gt_u.append(metadata.affordance_index[gt_tpg.terminals[i_spg+1].affordance[io]])
                    seg_pred_u.append(predicted_tpg.terminals[i_spg+1].affordance[io])

def infer(paths, gt_tpg, priors, grammar_dict, languages, duration):
    gt_subactivity, gt_objects, gt_affordance = get_ground_truth_label(gt_tpg)
    likelihoods = get_intermediate_results(paths, gt_tpg)
    obj_num = gt_objects.shape[0]

    # Segmentation
    # dp_start_time = time.time()
    trace_begin, trace_a, trace_o, trace_u, trace_s = dp_segmentation(priors, likelihoods)
    ttrace_begin, ttrace_a, ttrace_s = dp_segmentation_s(priors, likelihoods)
    # print('DP segmentation time elapsed: {}'.format(time.time() - dp_start_time))

    # Labels for evaluation
    seg_gt_s = list()
    seg_pred_s = list()
    seg_gt_u = list()
    seg_pred_u = list()

    gt_s = [list() for _ in range(duration)]
    pred_s = [list() for _ in range(duration)]
    gt_u = [list() for _ in range(duration)]
    pred_u = [list() for _ in range(duration)]

    for end_frame in range(1, int(trace_begin.shape[0])):
    # for end_frame in range(10, 350, 10):
    # for end_frame in [350]:
        # Gibbs sampling to refine the parsing
        tpg = generate_parse_graph(trace_begin, trace_a, trace_o, trace_u, trace_s, gt_tpg.terminals[0].start_frame, end_frame)
        # print str(gt_tpg), tpg_to_tokens(tpg, np.inf)
        # vizutil.visualize_tpg_labeling(gt_subactivity, gt_affordance, tpg, obj_num, end_frame)
        tpg.activity = gt_tpg.activity
        tpg = gibbs_sampling(tpg, grammar_dict, languages, priors, likelihoods)
        # vizutil.visualize_tpg_labeling(gt_subactivity, gt_affordance, tpg, obj_num, end_frame)

        # Prediction
        predicted_tpg = predict(grammar_dict, languages, tpg, end_frame, duration, priors, likelihoods)
        # vizutil.visualize_tpg_labeling(gt_subactivity, gt_affordance, predicted_tpg, obj_num, end_frame+duration)

        # Labels for evaluation
        get_next_subactivity_label(gt_tpg, predicted_tpg, seg_gt_s, seg_pred_s, end_frame)
        get_next_affordance_label(gt_tpg, predicted_tpg, seg_gt_u, seg_pred_u, obj_num, end_frame)

        subactivities, actions, objects, affordance = get_label(predicted_tpg, obj_num)
        pred_end_frame = np.min([subactivities.shape[0], gt_subactivity.shape[0], end_frame-1+duration])
        # print subactivities.shape, actions.shape, objects.shape, affordance.shape
        # print gt_subactivity.shape, gt_objects.shape, gt_affordance.shape
        for f in range(end_frame-1, pred_end_frame):
            gt_s[f-end_frame+1].append(gt_subactivity[f])
            pred_s[f-end_frame+1].append(subactivities[f])
            for io in range(obj_num):
                gt_u[f-end_frame+1].append(gt_affordance[io, f])
                pred_u[f-end_frame+1].append(affordance[io, f])

    print(gt_tpg.activity, tpg.activity, predicted_tpg.activity)
    print(str(gt_tpg))
    print(tpg_to_tokens(tpg, np.inf))

    final_pred_seg = np.zeros_like(seg_gt_s)
    for spg in predicted_tpg.terminals:
        final_pred_seg[spg.start_frame: spg.end_frame + 1] = spg.action

    # action_log_likelihood = np.load(os.path.join(paths.inter_root, 'likelihood', 'activity',
    #                                              gt_tpg.activity + '$' + gt_tpg.id + '.npy')).T
    # lstm_pred_s = np.argmax(action_log_likelihood, axis=0).tolist()
    print('Action detection micro evaluation:', utils.compute_accuracy(gt_s[0], pred_s[0], metric='macro'))
    print('Affordance detection micro evaluation:', utils.compute_accuracy(gt_u[0], pred_u[0], metric='macro'))
    return seg_gt_s, seg_pred_s, seg_gt_u, seg_pred_u, gt_s, pred_s, gt_u, pred_u, predicted_tpg.activity
    # return seg_gt_s, seg_pred_s, seg_gt_u, seg_pred_u, gt_s, pred_s, gt_u, pred_u, trace_s, lstm_pred_s, predicted_tpg.activity

def evaluate(paths):
    try:
        priors = load_prior(paths)
    except IOError:
        sys.exit('Prior information not found.')

    try:
        activity_corpus = pickle.load(open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'rb'))
    except IOError:
        sys.exit('Ground truth pickle file not found.')

    grammar_dict = grammarutils.read_induced_grammar(paths)
    languages = grammarutils.read_languages(paths)

    # Prediction duration
    duration = 45 + 1

    total_seg_gt_s = list()
    total_seg_pred_s = list()
    total_seg_gt_u = list()
    total_seg_pred_u = list()

    total_gt_s = [list() for _ in range(duration)]
    total_pred_s = [list() for _ in range(duration)]
    total_gt_u = [list() for _ in range(duration)]
    total_pred_u = [list() for _ in range(duration)]

    total_gt_e = list()
    total_pred_e = list()

    for activity, tpgs in activity_corpus.items():
        print(activity)
        for tpg in tpgs:
            print(tpg.id, tpg.terminals[-1].end_frame)
            # if tpg.subject != 'Subject5':
            #     continue
            # if tpg.id != '1204142858':  # Taking medicine, start_frame != 0
            #     continue
            # if tpg.id != '1204144736':
            #     continue
            # if tpg.id == '1204174554' or tpg.id == '1204142616' or tpg.id == '0510142336' or tpg.id == '1204175712' or tpg.id == '1130151154' or tpg.id == '0510172333' or tpg.id == '1130151154':
            #     continue
            infer_start_time = time.time()
            results = infer(paths, tpg, priors, grammar_dict, languages, duration)
            print('Inference time elapsed: {}s'.format(time.time() - infer_start_time))
            # seg_gt_s, seg_pred_s, seg_gt_u, seg_pred_u, gt_s, pred_s, gt_u, pred_u, trace_s, lstm_pred_s, e = results
            seg_gt_s, seg_pred_s, seg_gt_u, seg_pred_u, gt_s, pred_s, gt_u, pred_u, e = results

            # vizutils.plot_segmentation([gt_s[0], pred_s[0], trace_s[:-1], lstm_pred_s[:-1]], len(gt_s[0]))

            total_seg_gt_s.extend(seg_gt_s)
            total_seg_pred_s.extend(seg_pred_s)
            total_seg_gt_u.extend(seg_gt_u)
            total_seg_pred_u.extend(seg_pred_u)

            total_gt_e.append(metadata.activity_index[tpg.activity])
            total_pred_e.append(metadata.activity_index[e])

            for i in range(duration):
                total_gt_s[i].extend(gt_s[i])
                total_pred_s[i].extend(pred_s[i])
                total_gt_u[i].extend(gt_u[i])
                total_pred_u[i].extend(pred_u[i])

    print('===================  Frame wise Action detection  ===================')
    print(utils.compute_accuracy(total_gt_s[0], total_pred_s[0], metric='micro'))
    print(utils.compute_accuracy(total_gt_s[0], total_pred_s[0], metric='macro'))

    print('=================  Frame wise Affordance detection  =================')
    print(utils.compute_accuracy(total_gt_u[0], total_pred_u[0], metric='micro'))
    print(utils.compute_accuracy(total_gt_u[0], total_pred_u[0], metric='macro'))

    print('===================  Segment wise prediction  ===================')
    print(utils.compute_accuracy(total_seg_gt_s, total_seg_pred_s, metric='micro'))
    print(utils.compute_accuracy(total_seg_gt_s, total_seg_pred_s, metric='macro'))

    eval_gt_frame_s = list()
    eval_pred_frame_s = list()
    eval_gt_frame_u = list()
    eval_pred_frame_u = list()
    for i in range(duration):
        eval_gt_frame_s.extend(total_gt_s[i])
        eval_pred_frame_s.extend(total_pred_s[i])
        eval_gt_frame_u.extend(total_gt_u[i])
        eval_pred_frame_u.extend(total_pred_u[i])
    print('===================  Frame wise action prediction  ===================')
    print(utils.compute_accuracy(eval_gt_frame_s, eval_pred_frame_s, metric='micro'))
    print(utils.compute_accuracy(eval_gt_frame_s, eval_pred_frame_s, metric='macro'))

    print('===================  Frame wise affordance prediction  ===================')
    print(utils.compute_accuracy(eval_gt_frame_u, eval_pred_frame_u, metric='micro'))
    print(utils.compute_accuracy(eval_gt_frame_u, eval_pred_frame_u, metric='macro'))




def test(paths):
    activity_corpus = pickle.load(open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'rb'))

    spg_count = 0
    for activity, tpgs in activity_corpus.items()[:]:
        for tpg in tpgs:
            print(tpg)

            # if tpg.subject != 'Subject1':
            #     continue
            # spg_count += len(tpg.terminals)

    # print spg_count


def main():
    paths = config.Paths()
    start_time = time.time()
    np.random.seed(0)
    evaluate(paths)
    # test(paths)
    print('Time elapsed: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
