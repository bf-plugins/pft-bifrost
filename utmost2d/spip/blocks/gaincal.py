# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import

import bifrost as bf
from bifrost.pipeline import TransformBlock
from bifrost.DataType import DataType

from copy import deepcopy
import numpy as np

INT_CMULT_KERNEL = """
// Compute b = w * a

Complex<float> a_cf;
a_cf.real = a.real;
a_cf.imag = a.imag;

Complex<float> w_cf;
w_cf.real = w.real;
w_cf.imag = w.imag;

Complex<float> b_cf;
b_cf = a_cf * w_cf;

b.real = (int) b_cf.real;
b.imag = (int) b_cf.imag;
"""

FLOAT_CMULT_KERNEL = """
// Compute b = w * a

Complex<float> a_cf;
a_cf.real = a.real;
a_cf.imag = a.imag;

Complex<float> w_cf;
w_cf.real = w.real;
w_cf.imag = w.imag;

b = a_cf * w_cf;
"""

class ApplyWeightsBlock(TransformBlock):
    def __init__(self, iring, weights_callback=None, update_frequency=100,
                 output_dtype='cf32', *args, **kwargs):
        super(ApplyWeightsBlock, self).__init__(iring, *args, **kwargs)
        self.weights_callback = weights_callback
        self.update_frequency = update_frequency
        self.on_data_cnt = 0
        
        # Slightly different kernels for float / int output
        self.output_dtype = output_dtype
        assert output_dtype in ('cf32', 'ci8', 'ci16', 'ci32')
        if output_dtype == 'cf32':
            self.kernel = FLOAT_CMULT_KERNEL
        else:
            self.kernel = INT_CMULT_KERNEL

    def define_valid_input_spaces(self):
        """Return set of valid spaces (or 'any') for each input"""
        return ('cuda',)
    
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        ohdr = deepcopy(ihdr)
        
        # Send ihdr to weights generator class
        self.weights_callback.init(ihdr)

        # Set output dtype and make sure right kernel is in use        
        ohdr['_tensor']['dtype'] = self.output_dtype
        
        return ohdr

    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        
        # Update weights 
        if self.on_data_cnt % self.update_frequency == 0:
            weights = self.weights_callback.generate_weights()
        bf.map(self.kernel, {'a': idata, 'b': odata, 'w': weights})
        self.on_data_cnt += 1

class WeightsGenerator(object):
    
    def init(self, ihdr):
        """ Initialize by reading sequence header """
        self.itensor = ihdr["_tensor"]
        self.weights_cpu = bf.ndarray(np.ones_like(self.itensor["shape"]))
        self.weights_gpu = self.weights_cpu.copy('cuda')
    
    def generate_weights(self):
        """ Main function call that is required for weight generation """
        return self.weights_gpu


def apply_weights(iring, weights_callback, update_frequency=100, *args, **kwargs):
    """ Apply complex gain calibrations to antennas
    Args:
        iring (Ring or Block): Input data source.
        weights_callback: WeightsGenerator class that generates weights.  
                          Should return a bf.ndarray in CUDA space with the same shape as ispan.data
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.
   Returns:
        ApplyWeightsBlock: A new block instance.
    """
    return ApplyWeightsBlock(iring, weights_callback, update_frequency, *args, **kwargs)


