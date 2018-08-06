#!/usr/bin/env python3
from fractions import Fraction
from math import floor
"""
for n in range(0,100):
    curr = Fraction(1,1)
    for m in range(1,n):
        ratio = Fraction(m,n);
        curr = curr + (ratio**m)/(ratio**m-ratio**n)
    print("The value for " + str(n) + " is " +str(curr));
"""

for n in range(1,5000):
    curr_mine = 1
    curr_gs = 1
    for m in range(1,n):
        ratio = m/n;
        curr_mine = curr_mine + (ratio**m)/(ratio**m-ratio**n)
        curr_gs = curr_gs + n/(n-m)

    print(str(n) + "," +str(curr_mine) +","+str(curr_gs)+","+str(curr_gs/curr_mine));
