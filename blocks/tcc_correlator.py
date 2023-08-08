import time
import bifrost as bf
import btcc
from datetime import datetime
from copy import deepcopy
from pprint import pprint

#Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
#Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];

class TensorCoreCorrelator(bf.pipeline.TransformBlock):
    def __init__(self, iring, *args, **kwargs):
        super(TensorCoreCorrelator, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0

    def on_sequence(self, iseq):

        ihdr = iseq.header
        itensor = ihdr['_tensor']

        # initialise the tensor core correlator
        ichan = itensor['labels'].index('freq')
        iant  = itensor['labels'].index('station')
        ipol  = itensor['labels'].index('pol')
        itime = itensor['labels'].index('sample') 
        iblk  = itensor['labels'].index('tcc_block')

        nchan = itensor['shape'][ichan]
        nant  = itensor['shape'][iant]
        npol  = itensor['shape'][ipol]
        ntime = itensor['shape'][itime] 
        nblk  = itensor['shape'][iblk]

        nbits = int(itensor['dtype'].strip('ci'))

        self.tcc = btcc.Btcc()
        self.tcc.init(nbits, ntime*nblk, nchan, nant, npol)

        # Generate output tensor 
        ohdr = deepcopy(ihdr)
        ohdr['_tensor'] = {
            'dtype': 'ci32',
            'shape': [-1, nchan, nant*(nant+1)//2, npol, npol],
            'labels': ['time', 'freq', 'baseline', 'pol_i', 'pol_j'],
            'units': [itensor['units'][itime], itensor['units'][ichan], '', ''],
            'scales': [itensor['scales'][itime], itensor['scales'][ichan], [0, 1], [0, 1]]
        }
        pprint(ihdr)
        pprint(ohdr)
        return ohdr

    def on_data(self, ispan, ospan):
        now = datetime.now()
        #print(f"[{now}] {ispan.data.shape} {ispan.data.dtype} | {ospan.data.shape} {ospan.data.dtype}")
        self.tcc.execute(ispan.data[0], ospan.data[0], True)
        
def tensor_core_correlator(iring, *args, **kwargs):
    """ Tensor core correlator block 
    
    Tensor semantics:
        [time, freq, block, station, pol, sample]
    """
    return TensorCoreCorrelator(iring, *args, **kwargs)