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

class SquareBlock(TransformBlock):
    def __init__(self, iring, axis=None,
                 *args, **kwargs):
        super(SquareBlock, self).__init__(iring, *args, **kwargs)
        self.specified_axis = axis
    
    def define_valid_input_spaces(self):
        """Return set of valid spaces (or 'any') for each input"""
        return ('cuda',)
    
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        itype = DataType(itensor['dtype'])
        self.axis = self.specified_axis
        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']
        otensor['dtype'] = itype.as_floating_point()
        return ohdr

    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        bf.map("b = a * a", {'a': idata, 'b': odata})

def square(iring, axis=None, *args, **kwargs):
    """ Square an input (b = a*a)
    Args:
        iring (Ring or Block): Input data source.
        axis: Integer or string specifying the polarization axis. Defaults to
                'pol'. Not used if mode = 'scalar'.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.
   Returns:
        SquareBlock: A new block instance.
    """
    return SquareBlock(iring, axis, *args, **kwargs)


