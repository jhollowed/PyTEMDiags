# Joe Hollowed
# University of Michigan 2023
#
# Providing a set of utility functions for use by tem_diagnostics.py


# =========================================================================
    

from timeit import default_timer


# -------------------------------------------------------------------------


class logger:
    def __init__(debug, name='PyTEM'):
        self.debug = debug
        self.name  = name
        self.timer_running = False
    def print(s, with_timer=False):
        if(self.debug): print('({} debug) {}'.format(self.name, s))
        if(with_timer):
            self.timer()
    def timer(start_silent=True):
        if not self.timer_running:
            self.start = default_timer()
            self.timer_running = True
            if not start_silent(): self.print('timer started')
        else:
            self.stop = default_timer()
            self.timer_running = False
            self.print('elapsed time: {}'.format(self.stop - self.start))
