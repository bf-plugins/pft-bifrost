import time
from pprint import pprint
import bifrost as bf
from datetime import datetime

class PrintStuffBlock(bf.pipeline.SinkBlock):
    def __init__(self, iring, n_gulp_per_print=128, print_on_data=True, *args, **kwargs):
        super(PrintStuffBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0
        self.n_gulp_per_print = n_gulp_per_print
        self.print_on_data = print_on_data

    def on_sequence(self, iseq):
        print("[%s]" % datetime.now())
        print(iseq.name)
        #pprint(iseq.header)i
        pprint(iseq.header)
        self.n_iter = 0

    def on_data(self, ispan):
        if self.n_iter % self.n_gulp_per_print == 0 and self.print_on_data:
            now = datetime.now()
            d = ispan.data
            #d = np.array(d).astype('float32')
            print("[%s] %s %s" % (now, str(ispan.data.shape), str(ispan.data.dtype)))
        self.n_iter += 1

def print_stuff_block(iring, *args, **kwargs):
    """ Print Stuff! For debugging. """
    return PrintStuffBlock(iring, *args, **kwargs)