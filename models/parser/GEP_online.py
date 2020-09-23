"""
Created on Jan 25, 2018

@author: Siyuan Qi

Description of the file.

"""

import heapq

import numpy as np
import models.parser.grammarutils as grammarutils
import nltk.grammar
from tqdm import tqdm


def format_num(num):
    if num > 1e-3 or num == 0:
        return '{:.3f}'.format(num)
    else:
        return '{:.1e}'.format(num)


class State(object):
    def __init__(self, r, dot, start, end, i, j, rule_index, operation, last_i, last_j, last_rule_index, prefix, prob, forward, inner):
        self._r = r
        self._dot = dot
        self._start = start
        self._end = end

        # The parent rule is indexd by state_set[i][j][rule_index]
        self._i = i
        self._j = j
        self._rule_index = rule_index
        self._operation = operation

        self._last_i = last_i
        self._last_j = last_j
        self._last_rule_index = last_rule_index

        self._prefix = prefix
        self._prob = prob

        # Stockle, A. An Efficient Probabilistic Context-Free Parsing Algorithm that Computes Prefix Probabilities. (1995)
        # https://www.aclweb.org/anthology/J95-2002
        # In probability range
        self._forward = forward
        self._inner = inner

    def is_complete(self):
        return self._dot == len(self._r.rhs())

    def next_symbol(self):
        if self.is_complete():
            return None
        return self._r.rhs()[self._dot]

    def earley_equivalent(self, other_state):
        return self.earley_hash() == other_state.earley_hash()

    def earley_hash(self):
        rhs = [str(n) for n in self._r.rhs()]
        return '[{}:{}:{}] {} -> {}: {}'.format(self._dot, self._start, self._end, self._r.lhs(), rhs, self.prefix_str())

    def prefix_str(self):
        return ' '.join(self._prefix)

    def __repr__(self):
        rhs = [str(n) for n in self._r.rhs()]
        rhs = ' '.join(rhs[:self._dot]) + ' * ' + ' '.join(rhs[self._dot:])
        return '{} -> {} : {:.3f} ``{}" (start: {} end: {}) (forward:{}, inner:{}, {} from [{}:{}:{}])'\
            .format(self._r.lhs(), rhs, self._prob, ' '.join(self._prefix), self.start, self.end, self._forward, self._inner,
                    self._operation, self._i, self._j, self._rule_index)

    def tex(self, state_idx, prefix_tex, state_sets):
        if str(self._r.lhs()) == 'GAMMA':
            lhs = '\\Gamma'
        else:
            lhs = str(self._r.lhs())
        rhs = [str(n) for n in self._r.rhs()]
        rhs = ' '.join(rhs[:self._dot]) + ' \\boldsymbol{\\cdot} ' + ' '.join(rhs[self._dot:])
        rule = lhs + ' \\rightarrow ' + rhs

        if self._operation == 'root':
            comment = 'start rule'
        elif self._operation == 'predict':
            comment = 'predict: ({})'.format(self._rule_index)
        elif self._operation == 'scan':
            comment = 'scan: S({}, {})({})'.format(self._last_i, self._last_j, self._last_rule_index)
        elif self._operation == 'complete':
            comment = 'complete: ({}) and S({}, {})({})'.format(self._last_rule_index[1], self._last_i, self._last_j, self._last_rule_index[0])

        return '({}) & ${}$ & {} & {} & ``${}$" & {}\\\\'.format(state_idx, rule, format_num(self._forward), format_num(self._inner), prefix_tex, comment)

    @property
    def r(self): return self._r

    @property
    def dot(self): return self._dot

    @property
    def start(self): return self._start

    @property
    def end(self): return self._end

    @property
    def i(self): return self._i

    @property
    def j(self): return self._j

    @property
    def rule_index(self): return self._rule_index

    @property
    def prefix(self): return self._prefix

    @property
    def prob(self): return self._prob

    @property
    def forward(self): return self._forward

    @property
    def inner(self): return self._inner

    @prob.setter
    def prob(self, value):
        self._prob = value

    @forward.setter
    def forward(self, value):
        self._forward = value

    @inner.setter
    def inner(self, value):
        self._inner = value

class GeneralizedEarley(object):
    def __init__(self, grammar, class_num, mapping=None):
        self._grammar = grammar
        self._class_num = class_num
        self._classifier_output = np.empty((0, self._class_num))
        self._total_frame = self._classifier_output.shape[0]
        self._cached_log_prob = dict()
        self._cached_log_prefix_prob = dict()
        self._cached_grammar_prob = dict()
        self._cached_log_prob[''] = np.ones(self._total_frame) * np.finfo('d').min
        self._cached_log_prefix_prob[''] = 0.0
        self._cached_grammar_prob[''] = 1.0
        self._state_set = None
        self._queue = None
        self._max_log_prob = None
        self._best_l = None
        self._mapping = mapping
        self._state_set_id = {'': (0, 0)}
        # grammarutils.grammar_to_dot(self._grammar, '/media/hdd/home/baoxiong/grammar.txt')

    def parse_init(self):
        self._queue = []
        self._state_set = [[[]]]
        for r in self._grammar.productions():
            if str(r.lhs()) == 'GAMMA':
                self._state_set[0][0].append(State(r, 0, 0, 0, 0, 0, -1, 'root', 0, 0, 0, [], 0.0, 1.0, 1.0))
                break
        heapq.heappush(self._queue, ((1.0 - 1.0, (0, 0, '', self._state_set[0][0]))))

        self._max_log_prob = -np.inf
        self._best_l = None

    def state_set_vis(self):
        for m, m_set in enumerate(self._state_set):
            print('======================================================')
            for n, mn_set in enumerate(m_set):
                prefix_str = mn_set[0].prefix_str()
                for state_idx, state in enumerate(mn_set):
                    print('[{} {} / {}] {}, prior: {}, prefix: {}, parsing: {}'.format(
                                                        m, n, state_idx, state, self._cached_grammar_prob[prefix_str],
                                                        np.exp(self._cached_log_prefix_prob[prefix_str]),
                                                        np.exp(self._cached_log_prob[prefix_str][self._total_frame-1])
                                                    ))
            print('======================================================')

    def state_set_tex(self):
        print('\\begin{tabular}{|c|l|l|l|l|l|}\n\\hline\nstate \\# & rule & $\\mu$ & $\\nu$ & prefix & comment \\\\\n\\hline')
        for m, m_set in enumerate(self._state_set):
            for n, mn_set in enumerate(m_set):
                prefix_str = mn_set[0].prefix_str()
                prefix_tex = prefix_str or '\\epsilon'

                grammar_prior = format_num(self._cached_grammar_prob[prefix_str])
                parsing_prob = format_num(np.exp(self._cached_log_prob[prefix_str][self._total_frame-1]))
                prefix_prob = format_num(np.exp(self._cached_log_prefix_prob[prefix_str]))
                print('\\multicolumn{{6}}{{l}}{{$S({}, {}): l=``{}", p(l|G)={}, p(l|x, G)={}, p(l_{{\\cdots}}|x, G)={}$}} \\\\'
                      .format(m, n, prefix_tex, grammar_prior, parsing_prob, prefix_prob))
                print('\\hline')
                for state_idx, state in enumerate(mn_set):
                    print(state.tex(state_idx, prefix_tex, self._state_set))
                print('\\hline')
        print('\\multicolumn{5}{l}{Final output: $l^{*} = ``0 + 1"$ with probability 0.054} \\\\\n\\end{tabular}')

    def cached_prob_tex(self):
        print('\\begin{{tabular}}{{|{}|}}'.format('|'.join(['c']*(len(self._cached_log_prob.keys())+1))))
        print('\\hline')
        prefices = self._cached_log_prob.keys()[:]
        prefices.sort(key=lambda item: (len(item), item))
        print('Frame & $\\epsilon$' + ' & '.join(prefices) + ' \\\\')
        print('\\hline')
        for f in range(self._total_frame+1):
            print('{} & '.format(f) + ' & '.join([format_num(np.exp(self._cached_log_prob[prefix][f])) for prefix in prefices]) + ' \\\\')
        print('\\hline')
        print('\\end{tabular}')

    def debug(self, verbose=False):
        if verbose:
            self.state_set_vis()
            for l, p in self._cached_log_prob.items():
                print(l, p, self._cached_log_prefix_prob[l])

    def parse(self):
        self.parse_init()

        count = 0
        searched_prefices = set()
        while self._queue:
            # count += 1
            # if count % 100 == 0:
            #     tqdm.write('count {}'.format(count))
            _, (m, n, set_l, current_set) = heapq.heappop(self._queue)
            # tqdm.write(set_l)
            branch_log_probs = dict()
            # branch_log_probs[set_l] previously have the prefix probability of "set_l" string
            # since we are expanding this path on the prefix tree, we update this probability to parsing probability
            branch_log_probs[set_l] = self._cached_log_prob[set_l][self._total_frame - 1]
            if self._cached_log_prob[set_l][self._total_frame - 1] > self._max_log_prob:
                self._max_log_prob = self._cached_log_prob[set_l][self._total_frame - 1]
                self._best_l = set_l

            # self.state_set_vis()
            # TODO: new_scanned_states naturally satisfies set properties, assertion needed
            new_scanned_states = list()
            for rule_index, s in enumerate(current_set):
                if s.is_complete():
                    self.complete(m, n, rule_index, s)
                elif nltk.grammar.is_nonterminal(s.next_symbol()):
                    self.predict(m, n, rule_index, s)
                elif nltk.grammar.is_terminal(s.next_symbol()):
                    if m == self._total_frame:
                        continue
                    new_scanned_states.append(self.scan(m, n, rule_index, s))
                else:
                    raise ValueError('No operation (predict, scan, complete) applies to state {}'.format(s))

            for new_m, new_n, new_prefix_str in new_scanned_states:
                new_prefix = self._state_set[new_m][new_n][0].prefix

                # These states should all be after operation SCAN
                if not new_prefix_str in searched_prefices:
                    self._cached_grammar_prob[new_prefix_str] = 0
                    for new_s in self._state_set[new_m][new_n]:
                        self._cached_grammar_prob[new_prefix_str] += new_s.forward
                    prob = self.compute_prob(new_prefix)
                    branch_log_probs[new_prefix_str] = self._cached_log_prefix_prob[set_l]
                    for new_s in self._state_set[new_m][new_n]:
                        new_s.prob = prob
                    heapq.heappush(self._queue, (1.0 - prob, (new_m, new_n, new_prefix_str, self._state_set[new_m][new_n])))
                    searched_prefices.add(new_prefix_str)

            # Early stop
            if self._queue:
                best_prefix_string = self._queue[0][1][2]
                max_prefix_log_prob = self._cached_log_prefix_prob[best_prefix_string]
            else:
                max_prefix_log_prob = -np.inf
            max_branch_log_prob = max([val for key, val in branch_log_probs.items()])
            if branch_log_probs[set_l] == max_branch_log_prob:
                if self._max_log_prob > max_prefix_log_prob:
                    # tqdm.write('Find best parse before exhausting all strings.')
                    self.debug()
                    return self._best_l, self._max_log_prob

        # tqdm.write(', '.join(['{}: {}'.format(key, val) for key, val in self._cached_grammar_prob.items()]))
        self.debug()
        return self._best_l, self._max_log_prob

    def get_log_prob_sum(self):
        log_prob = np.log(self._classifier_output).transpose()
        log_prob_sum = np.zeros((self._class_num, self._total_frame, self._total_frame))
        for c in range(self._class_num):
            for b in range(self._total_frame):
                log_prob_sum[c, b, b] = log_prob[c, b]
        for c in range(self._class_num):
            for b in range(self._total_frame):
                for e in range(b+1, self._total_frame):
                    log_prob_sum[c, b, e] = log_prob_sum[c, b, e-1] + log_prob[c, e]
        return log_prob, log_prob_sum

    def compute_labels(self):
        log_prob, log_prob_sum = self.get_log_prob_sum()

        tokens = [int(token) for token in self._best_l.split(' ')]
        dp_tables = np.zeros((len(tokens), self._total_frame))
        traces = np.zeros_like(dp_tables)

        for end in range(0, self._total_frame):
            dp_tables[0, end] = log_prob_sum[tokens[0], 0, end]

        for token_i, token in enumerate(tokens):
            if token_i == 0:
                continue
            for end in range(token_i, self._total_frame):
                max_log_prob = -np.inf
                for begin in range(token_i, end+1):
                    check_prob = dp_tables[token_i-1, begin-1] + log_prob_sum[token, begin, end]
                    if check_prob > max_log_prob:
                        max_log_prob = check_prob
                        traces[token_i, end] = begin-1
                dp_tables[token_i, end] = max_log_prob

        # Back tracing
        token_pos = [-1 for _ in tokens]
        token_pos[-1] = self._total_frame - 1
        for token_i in reversed(range(len(tokens)-1)):
            token_pos[token_i] = int(traces[token_i+1, token_pos[token_i+1]])

        labels = - np.ones(self._total_frame).astype(np.int)
        labels[:token_pos[0]+1] = tokens[0]
        for token_i in range(1, len(tokens)):
            labels[token_pos[token_i-1]+1:token_pos[token_i]+1] = tokens[token_i]

        return labels, self._best_l.split(' '), token_pos

    def future_predict(self, epsilon=1e-10):
        pred_prob = np.ones(self._class_num) * epsilon
        m, n = self._state_set_id[self._best_l]
        next_symbol_available = False
        for state in self._state_set[m][n]:
            if (not state.is_complete()) and nltk.grammar.is_terminal(state.next_symbol()):
                pred_prob[self._mapping[state.next_symbol()]] += state.forward
                next_symbol_available = True
        if not next_symbol_available:
            pred_prob[self._mapping[self._best_l.split()[-1]]] = 1.0
        pred_prob /= np.sum(pred_prob)
        return pred_prob

    def complete(self, m, n, rule_index, s):
        # if s.rule_index == -1:
        #     return
        # back_s = self._state_set[s.i][s.j][s.rule_index]
        for back_s in self._state_set[s.i][s.j]:
            if str(back_s.next_symbol()) == str(s.r.lhs()) and back_s.end == s.start:
                forward_prob = back_s.forward * s.inner
                inner_prob = back_s.inner * s.inner
                # TODO: Check about this rule's    /complete operation
                new_s = State(back_s.r, back_s.dot + 1, back_s.start, s.end, back_s.i, back_s.j, back_s.rule_index,
                              'complete', s.i, s.j, (s.rule_index, rule_index), s.prefix, s.prob, forward_prob, inner_prob)

                # Stockle, A. 1995 p176 completion probability calculation
                state_exist = False
                for r_idx, exist_s in enumerate(self._state_set[m][n]):
                    if exist_s.earley_equivalent(new_s):
                        assert (not state_exist), 'Complete duplication'
                        state_exist = True
                        exist_s.forward += forward_prob
                        exist_s.inner += inner_prob

                if not state_exist:
                    self._state_set[m][n].append(new_s)

    def predict(self, m, n, rule_index, s):
        expand_symbol = str(s.next_symbol())
        for r in self._grammar.productions():
            production_prob = r.prob()
            forward_prob = s.forward * production_prob
            inner_prob = production_prob

            if expand_symbol == str(r.lhs()):
                new_s = State(r, 0, s.end, s.end, m, n, rule_index, 'predict', m, n, rule_index, s.prefix, s.prob, forward_prob, inner_prob)

                # Stockle, A. 1995 p176 prediction probability calculation
                state_exist = False
                for exist_s in self._state_set[m][n]:
                    if exist_s.earley_equivalent(new_s):
                        assert(not state_exist), 'Prediction duplication'
                        state_exist = True
                        exist_s.forward += forward_prob

                if not state_exist:
                    self._state_set[m][n].append(new_s)

    def scan(self, m, n, rule_index, s):
        new_prefix = s.prefix[:]
        new_prefix.append(str(s.next_symbol()))

        forward_prob = s.forward
        inner_prob = s.inner

        # TODO: check scan rule father index
        new_s = State(s.r, s.dot + 1, s.start, s.end + 1, s.i, s.j, s.rule_index, 'scan', m, n, rule_index, new_prefix, 0.0, forward_prob, inner_prob)
        # print(new_s)

        if m == len(self._state_set) - 1:
            new_n = 0
            self._state_set.append([])
        else:
            new_n = len(self._state_set[m + 1])

        # To eliminate same prefix branches
        state_exist = False
        new_prefix_str = new_s.prefix_str()
        for s_idx, state_set in enumerate(self._state_set[m + 1]):
            exist_s = state_set[0]
            # print('\n\n\n\n\n\n\n\n\n')
            # print(exist_s)
            # print(new_s)
            assert(not exist_s.earley_equivalent(new_s)), 'No same Earley state should appear for non-recursive grammar'
            if exist_s.prefix_str() == new_prefix_str:
                self._state_set[m + 1][s_idx].append(new_s)
                return m + 1, s_idx, new_prefix_str

        # print 'scan: S[{}, {}]'.format(m+1, new_n), new_s
        self._state_set[m + 1].append([])
        self._state_set[m + 1][new_n].append(new_s)
        self._state_set_id[new_prefix_str] = (m + 1, new_n)
        return m + 1, new_n, new_prefix_str

    def split_prefix(self, prefix):
        l = ' '.join(prefix)
        l_minus = ' '.join(prefix[:-1])
        if self._mapping:
            k = int(self._mapping[prefix[-1]])
        else:
            k = int(prefix[-1])
        return l, l_minus, k

    def update_prob(self, new_frame_prob):
        self._classifier_output = np.vstack([self._classifier_output, new_frame_prob])
        prev_total_frame = self._total_frame
        self._total_frame += 1

        for l in sorted(self._cached_log_prob.keys()):
            if l == '':
                self._cached_log_prob[l] = np.hstack([self._cached_log_prob[l], np.finfo('d').min])
            else:
                prefix = l.split()
                l, l_minus, k = self.split_prefix(prefix)


                # Update parsing probability
                transition_log_prob = np.log(self._cached_grammar_prob[l]) - np.log(self._cached_grammar_prob[l_minus])
                max_log = max(self._cached_log_prob[l][prev_total_frame-1],
                              self._cached_log_prob[l_minus][prev_total_frame-1] + transition_log_prob)

                log_prob = np.log(new_frame_prob[k]) + max_log + \
                    np.log(np.exp(self._cached_log_prob[l][prev_total_frame-1] - max_log) +
                    np.exp(self._cached_log_prob[l_minus][prev_total_frame-1] + transition_log_prob - max_log))
                self._cached_log_prob[l] = np.hstack([self._cached_log_prob[l], log_prob])

                # TODO: update self._cached_log_prefix_prob
                # Update prefix probability
                max_log = max(self._cached_log_prob[l][0],
                              np.max(self._cached_log_prob[l_minus] + transition_log_prob))

                self._cached_log_prefix_prob[l] = np.exp(self._cached_log_prefix_prob[l] - max_log)
                self._cached_log_prefix_prob[l] += new_frame_prob[k] * np.exp(self._cached_log_prob[l_minus][self._total_frame - 2] + transition_log_prob - max_log)
                self._cached_log_prefix_prob[l] = np.log(self._cached_log_prefix_prob[l]) + max_log


    def compute_prob(self, prefix):
        l, l_minus, k = self.split_prefix(prefix)

        # Store grammar transition probability
        # Assume l = {l_minus, k}
        # self._cached_grammar_log_prob[l] = log(p(l | G))
        # transtition_log_prob = log( p(l... | G) / p(l_minus... | G)) = log(p(k | l_minus, G))
        transition_log_prob = np.log(self._cached_grammar_prob[l]) - np.log(self._cached_grammar_prob[l_minus])
        # transition_log_prob = 0

        if l not in self._cached_log_prob:
            # Initialize p(l|x_{0:T}) to negative infinity
            self._cached_log_prob[l] = np.ones(self._total_frame) * np.finfo('d').min
            self._cached_log_prefix_prob[l] = np.finfo('d').min
            if len(prefix) == 1:
                # Initializiation for p(l|x_{0:T}) when T = 0 and l only contains one symbol
                self._cached_log_prob[l][0] = np.log(self._classifier_output[0, k]) + transition_log_prob

            # Compute p(l)
            for t in range(1, self._total_frame):
                # To prevent numerical underflow for np.exp when the exponent is too small
                # In the meanwhile, prevent overflow for np.exp(maximum - regularizer)

                max_log = max(self._cached_log_prob[l][t - 1],
                              self._cached_log_prob[l_minus][t - 1] + transition_log_prob)

                # p(l|x_{0:t}, G) = y_t^{k} (p(l | x_{0:t-1}, G) + p(k | l^{-}, G) p(l^{-} | x_{0:t-1}, G))

                self._cached_log_prob[l][t] \
                    = np.log(self._classifier_output[t, k]) + max_log + \
                    np.log(np.exp(self._cached_log_prob[l][t - 1] - max_log) +
                    np.exp(self._cached_log_prob[l_minus][t - 1] + transition_log_prob - max_log))


            # Compute p(l...)
            if self._total_frame == 1:
                # When only 1 frame, p(l...|x_{0:t}) = p(l... | x_0) = p(l | x_0)
                # self._cached_log_prob[l][self._total_frame] = self._cached_log_prob[l][0]
                self._cached_log_prefix_prob[l] = self._cached_log_prob[l][0]
            else:
                max_log = max(self._cached_log_prob[l][0],
                              np.max(self._cached_log_prob[l_minus] + transition_log_prob))

                # (ICML 2018) Generalized Earley parser Equation(3)
                # self._cached_log_prob[l][self._total_frame] = np.exp(self._cached_log_prob[l][0] - max_log)
                self._cached_log_prefix_prob[l] = np.exp(self._cached_log_prob[l][0] - max_log)

                for t in range(1, self._total_frame):
                    self._cached_log_prefix_prob[l] += self._classifier_output[t, k] * \
                                        np.exp(self._cached_log_prob[l_minus][t - 1] + transition_log_prob - max_log)

                self._cached_log_prefix_prob[l] = np.log(self._cached_log_prefix_prob[l]) + max_log

        # Search according to prefix probability (Prefix probability stored in the last dimension)
        # TODO: better to return log instead of exp
        # return np.exp(self._cached_log_prob[l][self._total_frame])
        return np.exp(self._cached_log_prefix_prob[l])


def main():
    pass


if __name__ == '__main__':
    main()