"""
Created on 5/1/19

@author: Baoxiong Jia

Description:

"""

import os
import subprocess
import tempfile


def main():
    project_path = '/media/hdd/home/baoxiong/Projects'
    breakfast_path = os.path.join(project_path, 'TPAMI2019', 'tmp', 'breakfast')
    corpus_dir = os.path.join(breakfast_path, 'corpus')
    grammar_dir = os.path.join(breakfast_path, 'grammar')
    madios_path = os.path.join(project_path, 'Tools', 'madios', 'build', 'madios')

    eta = 1
    alpha = 0.1
    context_size = 2
    coverage = 0.5

    if not os.path.exists(grammar_dir):
        os.makedirs(grammar_dir)

    for f in os.listdir(corpus_dir):
        corpus_path = os.path.join(corpus_dir, f)
        grammar_path = os.path.splitext(os.path.join(grammar_dir, f))[0] + '.pcfg'
        cmd = '{} {} {} {} {} {}'.format(madios_path, corpus_path, eta, alpha, context_size, coverage)

        grammar = False
        with open(grammar_path, 'w') as grammar_file:
            for line in os.popen(cmd).readlines():
                if grammar:
                    if line.strip() != '':
                        grammar_file.write(line)
                if line.startswith('Time'):
                    grammar = True
        print('Finishing {}'.format(corpus_path))
if __name__ == '__main__':
    main()
