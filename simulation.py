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

        numbers = [float(row.split()[0]) for row in data.split('\n')[:13]]

        self.n = int(numbers[0])
        self.m = numbers[1]
        self.epsilon = numbers[2]
        self.R = numbers[3]

        self.f = numbers[4]
        self.L = numbers[5]
        self.a = numbers[6]

        self.To = numbers[7]
        self.tau = numbers[8]
        self.s_o = int(numbers[9])
        self.s_d = int(numbers[10])
        self.s_out = int(numbers[11])
        self.s_xyz = int(numbers[12])

    def __str__(self):
        return "To_%s_tau_%s_n_%s" % (self.To, self.tau, self.n)


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

        potential_output = self.prepare_output_path(output_filename, 'potential')
        potential_reporter = Reporter(potential_output)

        self.reporters_to_params = (
            (potential_reporter, lambda: self.epsilon),
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