"""
Created on Feb 24, 2017

@author: Siyuan Qi

Description of the file.

"""

import collections
import os
import time
import itertools

import numpy as np
import nltk

import functools
import datasets.CAD.cad_config as config

def induce_activity_grammar(paths):
    """
    Parameters for grammar induction:
    <eta>:          threshold of detecting divergence in the RDS graph, usually set to 0.9
    <alpha>:        significance test threshold, usually set to 0.01 or 0.1
    <context_size>: size of the context window used for search for Equivalence Class, usually set to 5 or 4, a value
                    less than 3 means that no equivalence class can be found.
    <coverage>:     threhold for bootstrapping Equivalence classes, usually set to 0.65. Higher values will result in
                    less bootstrapping.

    :param paths: paths configuration of the project
    :return:
    """
    # TODO: experiment on the parameters
    adios_path = os.path.join(paths.project_root, 'src', 'cpp', 'madios', 'madios')
    eta = 1
    alpha = 0.1
    context_size = 4
    coverage = 0.5

    for corpus_filename in os.listdir(os.path.join(paths.tmp_root, 'corpus')):
        corpus_path = os.path.join(paths.tmp_root, 'corpus', corpus_filename)
        os.system('{} {} {} {} {} {}'.format(adios_path, corpus_path, eta, alpha, context_size, coverage))


def read_languages(paths):
    languages = dict()
    for corpus_filename in os.listdir(os.path.join(paths.tmp_root, 'corpus')):
        corpus_path = os.path.join(paths.tmp_root, 'corpus', corpus_filename)
        with open(corpus_path) as f:
            language = f.readlines()
            language = [s.strip().split() for s in language]
            language = [s[1:-1] for s in language]
            languages[os.path.splitext(corpus_filename)[0]] = language
    return languages


def get_pcfg(rules, index=False, mapping=None):
    root_rules = list()
    non_terminal_rules = list()
    grammar_rules = list()
    for rule in rules:
        tokens = rule.split()
        for i in range(len(tokens)):
            token = tokens[i]
            if token[0] == 'E':
                tokens[i] = tokens[i].replace('E', 'OR')
            elif token[0] == 'P':
                tokens[i] = tokens[i].replace('P', 'AND')
            elif index and mapping and token[0] == "'":
                tokens[i] = "'{}'".format(mapping[token.strip("'")])
        rule = ' '.join(tokens)

        if rule.startswith('S'):
            root_rules.append(rule)
        else:
            non_terminal_rules.append(rule)

    for k, v in collections.Counter(root_rules).items():
        grammar_rules.append(k + ' [{}]'.format(float(v) / len(root_rules)))
    grammar_rules.extend(non_terminal_rules)
    return grammar_rules

def read_grammar(filename, index=False, mapping=None, insert=True):
    with open(filename) as f:
        rules = [rule.strip() for rule in f.readlines()]
        if insert:
            rules.insert(0, 'GAMMA -> S [1.0]')
        grammar_rules = get_pcfg(rules, index, mapping)
        grammar = nltk.PCFG.fromstring(grammar_rules)
    return grammar

def read_induced_grammar(paths):
    # Read grammar into nltk
    grammar_dict = dict()
    for activity_grammar_file in os.listdir(os.path.join(paths.tmp_root, 'grammar')):
        with open(os.path.join(paths.tmp_root, 'grammar', activity_grammar_file)) as f:
            rules = [rule.strip() for rule in f.readlines()]
            grammar_rules = get_pcfg(rules)
            grammar = nltk.PCFG.fromstring(grammar_rules)
            grammar_dict[os.path.splitext(activity_grammar_file)[0]] = grammar
            # print activity_grammar_file
            # print grammar
    return grammar_dict


def get_production_prob(selected_edge, grammar):
    # Find the corresponding production rule of the edge, and return its probability
    for production in grammar.productions(lhs=selected_edge.lhs()):
        if production.rhs() == selected_edge.rhs():
            return production.prob()


def find_parents(selected_edge, edge_idx, chart):
    parents = []
    # Find the parent edges that lead to the selected edge
    for p_idx, p_edge in enumerate(chart.edges()):
        # Important: Note that p_edge.end() is not equal to p_edge.start() + p_edge.dot(),
        # when a node in the edge spans several tokens in the sentence
        if p_idx < edge_idx and p_edge.end() == selected_edge.start() and p_edge.nextsym() == selected_edge.lhs():
            parents.append((p_edge, p_idx))
    return parents


def get_edge_prob(selected_edge, edge_idx, chart, grammar, level=0):
    # print(''.join(['\t' for _ in range(level)]) + '------------ Edge {} ------------'.format(selected_edge))
    # Compute the probability of the edge by recursion
    prob = get_production_prob(selected_edge, grammar)
    parents = find_parents(selected_edge, edge_idx, chart)
    if len(parents) > 0:
        parent_prob = 0
        for parent_edge, p_idx in parents:
            # print(''.join(['\t' for _ in range(level)]) + 'Parent edge {}'.format(parent_edge))
            p_prob = get_edge_prob(parent_edge, p_idx, chart, grammar, level + 1)
            # print(''.join(['\t' for _ in range(level)]) + 'Parent prob {}'.format(p_prob))
            parent_prob +=  p_prob
        prob *= parent_prob
    # print(''.join(['\t' for _ in range(level)]) + 'Edge {} : probability {}'.format(selected_edge, prob))
    return prob


def remove_duplicate(tokens):
    return [t[0] for t in itertools.groupby(tokens)]


def compute_sentence_probability(grammar, tokens):
    invalid_prob = 1e-20

    earley_parser = nltk.EarleyChartParser(grammar, trace=0)
    viterbi_parser = nltk.ViterbiParser(grammar)
    try:
        e_chart = earley_parser.chart_parse(tokens)
    except ValueError:
        return 0
        # d, tokens = find_closest_tokens(language, tokens)
        # return invalid_prob ** d

    # If the sentence is complete, return the Viterbi likelihood
    v_parses = viterbi_parser.parse_all(tokens)
    if v_parses:
        prob = functools.reduce(lambda a, b: a+b.prob(), v_parses, 0)/len(v_parses)
        return prob

    # If the sentence is incomplete, return the sum of probabilities of all possible sentences
    prob = 0
    for edge in e_chart.edges():
        if edge.end() == len(tokens) and isinstance(edge.nextsym(), str):
            prob += get_edge_prob(edge, e_chart, grammar)
    return prob


def compute_grammar_prefix_probability(grammar, tokens):
    earley_parser = nltk.EarleyChartParser(grammar, trace=1)
    try:
        e_chart = earley_parser.chart_parse(tokens)
    except ValueError:
        return 0.0

    prob = 0
    # If the sentence is incomplete, return the sum of probabilities of all possible sentences
    for edge_idx, edge in enumerate(e_chart.edges()):
        if edge.end() == len(tokens) and len(edge.rhs()):
            # print('----------------------------')
            # print(edge)
            # print(edge.nextsym())
            if not edge.nextsym():
                # If the sentence is a valid complete sentence
                prob += get_edge_prob(edge, edge_idx, e_chart, grammar, level=0)
            elif isinstance(edge.nextsym(), str):
                # If the sentence is a valid prefix
                prob += get_edge_prob(edge, edge_idx, e_chart, grammar, level=0)
    return prob


def earley_predict(grammar, tokens):
    tokens = remove_duplicate(tokens)
    symbols = list()
    earley_parser = nltk.EarleyChartParser(grammar, trace=0)
    try:
        e_chart = earley_parser.chart_parse(tokens)
    except ValueError:
        return list()
    end_edges = list()

    for edge in e_chart.edges():
        # print edge
        if edge.end() == len(tokens):
            # Only add terminal nodes
            if isinstance(edge.nextsym(), str):
                symbols.append(edge.nextsym())
                end_edges.append(edge)

    probs = list()
    for edge_idx, end_edge in enumerate(end_edges):
        probs.append(get_edge_prob(end_edge, edge_idx, e_chart, grammar))

    # Eliminate duplicate
    symbols_no_duplicate = list()
    probs_no_duplicate = list()
    for s, p in zip(symbols, probs):
        if s not in symbols_no_duplicate:
            symbols_no_duplicate.append(s)
            probs_no_duplicate.append(p)
        else:
            probs_no_duplicate[symbols_no_duplicate.index(s)] += p

    return symbols_no_duplicate, probs_no_duplicate


def grammar_to_dot(grammar, filename):
    and_nodes = list()
    or_nodes = list()
    terminal_nodes = list()
    root_branch_count = 0

    edges = list()
    for production in grammar.productions():
        if production.prob() == 1:
            and_nodes.append(str(production.lhs()))
            for i, child_node in enumerate(production.rhs()):
                edges.append(str(production.lhs()) + ' -> ' + str(child_node) + u' [penwidth=3, weight=3, label={}]\n'.format(chr(9312+i)))
        else:
            or_nodes.append(str(production.lhs()))
            if str(production.lhs()) == 'S':
                root_branch_count += 1
                and_nodes.append('S{}'.format(root_branch_count))
                edges.append(str(production.lhs()) + ' -> ' + str('S{}'.format(root_branch_count)) + '[label = "' + "{0:.2f}".format(production.prob()) + '", penwidth=' + str(1. + 2.*production.prob()) + ', weight=3]\n')
                for i, child_node in enumerate(production.rhs()):
                    edges.append(str('S{}'.format(root_branch_count)) + ' -> ' + str(child_node) + u' [penwidth=3, weight=3, label={}]\n'.format(chr(9312+i)))
            else:
                for child_node in production.rhs():
                    edges.append(str(production.lhs()) + ' -> ' + str(child_node) + '[label = "' + "{0:.2f}".format(production.prob()) + '", penwidth=' + str(1. + 2.*production.prob()) + ', weight=3]\n')

        for child_node in production.rhs():
            if isinstance(child_node, str):
                terminal_nodes.append(child_node)

    vertices = list()
    and_nodes = set(and_nodes)
    or_nodes = set(or_nodes)
    terminal_nodes = set(terminal_nodes)

    for and_node in and_nodes:
        vertices.append(and_node + ' [shape=doublecircle, fillcolor=green, style=filled, color=blue, ranksep=0.5, nodesep=0.5]\n')
    for or_node in or_nodes:
        vertices.append(or_node + ' [shape=circle, fillcolor=yellow, style=filled, color=blue, ranksep=0.5, nodesep=0.5]\n')
    for terminal in terminal_nodes:
        vertices.append(terminal + ' [shape=box, fillcolor=white, style=filled, ranksep=0.5, nodesep=0.5]\n')

    # edges = set(edges)
    with open(filename, 'w') as f:
        f.write('digraph G {\nordering=out\n')
        #
        # for vertex in vertices:
        #     f.write(vertex)
        f.writelines(vertices)
        # for edge in edges:
        #     f.write(edge)
        f.writelines(edges)
        f.write('}')


def test(paths):
    grammar_dict = read_induced_grammar(paths)
    languages = read_languages(paths)

    #########################################
    # # Draw grammar
    # fig_folder = os.path.join(paths.tmp_root, 'figs', 'grammar')
    # ext = 'pdf'
    # if not os.path.exists(fig_folder):
    #     os.makedirs(fig_folder)
    # for task, grammar in grammar_dict.items():
    #     grammar = grammar_dict[task]
    #     dot_filename = os.path.join(fig_folder, '{}.dot'.format(task))
    #     fig_filename = os.path.join(fig_folder, '{}.{}'.format(task, ext))
    #     grammar_to_dot(grammar, dot_filename)
    #     os.system('dot -T{} {} > {}'.format(ext, dot_filename, fig_filename))
    #     # break

    # sentence = 'null reaching opening reaching moving cleaning moving placing reaching closing'

    # grammar = grammar_dict['stacking_objects']
    # sentence = 'null reaching'

    # grammar = grammar_dict['unstacking_objects']
    # language = languages['unstacking_objects']

    grammar = grammar_dict['microwaving_food']

    # sentence1 = 'null reaching moving placing reaching moving placing reaching moving'
    # sentence2 = 'null reaching moving placing reaching moving placing reaching moving placing'

    sentence1 = 'null reaching opening reaching moving placing reaching closing'
    sentence2 = 'null reaching opening reaching moving placing reaching closing null'

    # sentence1 = 'null reaching moving placing'
    # sentence2 = 'null reaching moving placing reaching'

    tokens1 = sentence1.split()
    tokens2 = sentence2.split()
    # d, matched_tokens = find_closest_tokens(language, tokens)
    # print(d, matched_tokens)
    # print(compute_sentence_probability(grammar, tokens))
    # print(earley_predict(grammar, tokens1))
    # print(compute_sentence_probability(grammar, tokens1))
    # print(compute_sentence_probability(grammar, tokens2))
    p_l_ = compute_grammar_prefix_probability(grammar, tokens1)
    p_l = compute_grammar_prefix_probability(grammar, tokens2)
    print(p_l_, p_l)
    print(p_l / p_l_)
    # print(predict_next_symbols(grammar, matched_tokens))

    # sentence = 'null reaching moving placing reaching moving placing reaching movin'
    # tokens = sentence.split()
    # print tokens
    # for activity, grammar in grammar_dict.items():
    #     language = languages[activity]
    #     d, matched_tokens = find_closest_tokens(language, tokens)
    #     print activity, d, matched_tokens
    #     print compute_sentence_probability(grammar, language, tokens)
    #     print predict_next_symbols(grammar, matched_tokens)

    #########################################
    # Draw prediction parse graph
    # grammar = grammar_dict['stacking_objects']
    # sentence = 'null reaching'
    # # sentence = 'null reaching moving placing reaching moving placing null reaching moving placing null'
    # # sentence = 'null reaching moving placing reaching moving placing'
    # tokens = sentence.split()
    # # print predict_next_symbols(grammar, tokens)
    # parse_tree = get_prediciton_parse_tree(grammar, tokens, 'tree.ps')


def main():
    paths = config.Paths()
    start_time = time.time()
    # induce_activity_grammar(paths)
    # read_induced_grammar(paths)
    test(paths)
    print('Time elapsed: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
