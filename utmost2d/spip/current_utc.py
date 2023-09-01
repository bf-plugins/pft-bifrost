#!/usr/bin/env python
from astropy.time import Time
from datetime import datetime

t = Time(datetime.utcnow()).iso
print(t.replace(" ", "-"))
