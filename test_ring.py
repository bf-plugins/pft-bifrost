from bifrost.ring2 import Ring, ring_view

r1 = Ring(space='system', name='r1', owner=None, core=1)
r2 = ring_view(r1, header_transform=None)

print(r1.base)
print(r2.base)

#This causes double free
#r1._destroy()
#r2._destroy()

# This causes double free / segfault
r1.__del__()
#r2.__del__()
