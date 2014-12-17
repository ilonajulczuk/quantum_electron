from __future__ import division
import sys
import os
import numpy as np


def save_to_file(results, filename, blank_lines=False):
    with open(filename, 'a+') as f:
        try:
            template = "{}\t" * len(results[0]) + "\n"
            for x in results:
                f.write(template.format(*list(x)))
        except TypeError:
            template = "{}\t\n"
            for x in results:
                f.write(template.format(x))
        if blank_lines:
            f.write('\n\n')


class Configuration(object):
    def __init__(self, filename):
        with open(filename) as f:
            data = f.read()

        numbers = [float(row.split()[0]) for row in data.split('\n')[:9]]
        self.delta_tau = numbers[0]
        self.N = int(numbers[1])
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
        self.blank_lines = blank_lines
        if os.path.exists(filename):
            os.remove(filename)

    def store(self, results):
        save_to_file(results, self.output_filename, self.blank_lines)

    def __str__(self):
        return 'Reporter %s' % self.output_filename


class Simulation(object):
    def __init__(self, configuration_file, output_filename=None):
        self.conf = Configuration(configuration_file)

        self.epsilon = 0
        self.tau = 0

        self.xs = np.linspace(0, 1, self.conf.N)

        self.delta_x = self.xs[1] - self.xs[0]

        self.psi_real = np.sqrt(2) * np.sin(np.pi * self.conf.n * self.xs)
        self.psi_imaginary = np.zeros(self.conf.N)
        self.H_r = self.compute_hamiltionian(self.psi_real)
        self.H_i = self.compute_hamiltionian(self.psi_imaginary)
        self.output_filename = output_filename

    def prepare_output_path(self, output_filename, param_name):
        dirs = "/".join(output_filename.split('/')[:-1])
        filename = output_filename.split('/')[-1]
        return dirs + '/' + param_name + '_' + str(self.conf) + filename

    def step(self):
        self.tau += self.conf.delta_tau
        psi_real_half = self.psi_real + self.H_i * self.conf.delta_tau / 2
        self.H_r = self.compute_hamiltionian(psi_real_half)
        self.psi_imaginary = self.psi_imaginary - self.H_r * self.conf.delta_tau

        self.H_i = self.compute_hamiltionian(self.psi_imaginary)
        self.psi_real = psi_real_half + self.H_i * self.conf.delta_tau / 2
        return self.psi_real, self.psi_imaginary

    def compute_system_parameters(self):
        self.N = self.delta_x * (np.dot(self.psi_imaginary, self.psi_imaginary) + np.dot(self.psi_real, self.psi_real))
        self.x = self.delta_x * np.dot(self.xs, (np.power(self.psi_imaginary, 2) + np.power(self.psi_real, 2)))

        self.epsilon = self.delta_x * (np.dot(self.psi_imaginary, self.H_i) + np.dot(self.psi_real, self.H_r))
        self.ro = np.dot(self.psi_imaginary, self.psi_imaginary) + np.dot(self.psi_real, self.psi_real)
        self.rox = self.psi_imaginary * self.psi_imaginary + self.psi_real * self.psi_real

    def compute_hamiltionian(self, psi):
        hamiltionian = np.zeros(self.conf.N)
        hamiltionian[1:-1] = (0.5 * (psi[1:-1] * 2 - psi[:-2] - psi[2:]) /
                              self.delta_x ** 2 + self.conf.kappa * self.xs[1:-1] * psi[1:-1] *
                              np.sin(self.conf.omega * self.tau))
        return hamiltionian

    def run(self, s_o=None, s_d=None, s_out=None, s_xyz=None):
        output_filename = self.output_filename
        self.reporters_to_params = []
        self.psi_real_reporter = Reporter(self.prepare_output_path(output_filename, 'psi_real'),
                                          blank_lines=True)
        self.psi_imaginary_reporter = Reporter(self.prepare_output_path(output_filename, 'psi_imaginary'),
                                               blank_lines=True)

        self.rox_reporter = Reporter(self.prepare_output_path(output_filename, 'rox'),
                                     blank_lines=True)
        params = ['epsilon', 'ro', 'N', 'x']

        for param in params:
            output = self.prepare_output_path(output_filename, param)
            reporter = Reporter(output)

            self.reporters_to_params.append(
                (reporter, param),
            )

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
                self.psi_real_reporter.store(self.psi_real)
                self.psi_imaginary_reporter.store(self.psi_imaginary)
            if j % s_out == 0:
                self.compute_system_parameters()
                self.rox_reporter.store(self.rox)
                for reporter, param in self.reporters_to_params:
                    reporter.store([[self.tau, getattr(self, param)]])


def main():
    configuration_file = sys.argv[1]
    output_file = sys.argv[2]

    if len(sys.argv) == 4 and int(sys.argv[3]):
        percents = [0.90, 0.92, 0.94, 0.98, 1, 1.02, 1.04, 1.08, 1.10]
    else:
        percents = [1]
    for x in percents:
        prefix, _, suffix = output_file.rpartition('_')
        percent = str(x)

        output_file_percent = prefix + percent + '_' + suffix
        simulation = Simulation(configuration_file, output_file_percent)
        simulation.conf.omega *= x
        simulation.run()


if __name__ == '__main__':
    main()