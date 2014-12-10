from __future__ import division
from math import sqrt, log
from random import random, choice
import sys

import numpy as np


def save_to_file(results, filename):
    with open(filename, 'w+') as f:
        template = "{}\t" * len(results[0]) + "\n"
        for x in results:
            f.write(template.format(*list(x)))


def append_to_file(results, filename, blank_lines=False):
    with open(filename, 'a+') as f:
        if blank_lines:
            f.write('\n\n')
        template = "{}\t" * len(results[0]) + "\n"
        for x in results:
            f.write(template.format(*list(x)))


class Configuration(object):
    def __init__(self, filename):
        with open(filename) as f:
            data = f.read()

        numbers = [float(row.split()[0]) for row in data.split('\n')[:9]]
        self.delta_tau = numbers[0]
        self.N = numbers[1]
        self.kappa = numbers[2]
        self.omega = numbers[3]
        self.n = numbers[4]

        self.s_o = int(numbers[5])
        self.s_d = int(numbers[6])
        self.s_out = int(numbers[7])
        self.s_xyz = int(numbers[8])
        self.tau = 0

    def __str__(self):
        return "n_%s_kappa_%s_w_%s" % (self.n, self.kappa, self.omega)


class Reporter(object):
    def __init__(self, filename=None, blank_lines=False):

        self.output_filename = filename
        self.output_file_created = False
        self.results_output_file_created = False
        self.blank_lines = blank_lines

    def store(self, results):
        if not self.output_file_created:
            save_to_file(results, self.output_filename)
            self.output_file_created = True
        else:
            append_to_file(results, self.output_filename, self.blank_lines)


class Simulation(object):
    def __init__(self, configuration_file, output_filename=None):
        self.conf = Configuration(configuration_file)

        self.epsilon = 0

        epsilon_output = self.prepare_output_path(output_filename, 'epsilon')
        epsilon_reporter = Reporter(epsilon_output)

        self.reporters_to_params = (
            (epsilon_reporter, lambda: self.epsilon),
        )

    def prepare_output_path(self, output_filename, param_name):
        dirs = "/".join(output_filename.split('/')[:-1])
        filename = output_filename.split('/')[-1]
        return dirs + '/' + param_name + '_' + str(self.conf) + filename

    def step(self):
        pass

    def run(self, s_o=None, s_d=None, s_out=None, s_xyz=None):
        if s_o is None:
            s_o = self.conf.s_o
        if s_d is None:
            s_d = self.conf.s_d
        if s_out is None:
            s_out = self.conf.s_out
        if s_xyz is None:
            s_xyz = self.conf.s_xyz

        for j in range(s_o):
            self.step()

        for j in range(1, s_d):
            self.step()

            if j % s_xyz == 0:
                pass
            if j % s_out == 0:
                pass
            if j % s_o == 0:
                for reporter, param in self.reporters_to_params:
                    reporter.store([[j * self.conf.tau, param()]])


def main():
    configuration_file = sys.argv[1]
    output_file = sys.argv[2]
    simulation = Simulation(configuration_file, output_file)
    simulation.run()


if __name__ == '__main__':
    main()