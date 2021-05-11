import SimBot
import random
import numpy as np

def SimpleAveragePick(sets, cycles, target = 0):
    picklist = list()
    for c in range(cycles):
        frm = [sets[n].iloc[a] for n, a in enumerate([random.randint(1,len(s)-1) for s in sets])]
        for n, f in enumerate(frm):
            print('---Choice '+str(n)+'---')
            print(f)
        pick = frm[int(input('Pick From Choices: '))]
        picklist.append(pick)
    return np.average(picklist, axis=0)