import BaseExtra
import random
import numpy as np

def SimpleAveragePick(target, sets, cycles):
    picklist = list()
    for c in range(cycles):
        frm = [sets[n].iloc[a] for n, a in enumerate([random.randint(1,len(s)-1) for s in sets])]
        pick = frm[BaseExtra.closestvector(target, frm)]
        picklist.append(pick)
    return np.average(picklist, axis=0)

def TriangleMove(target, sets, cycles):
    picklist = list()
    start = np.average([np.average(s, axis=0) for s in sets], axis=0)
    for c in range(cycles):
        frm = [sets[n].iloc[a] for n, a in enumerate([random.randint(1,len(s)-1) for s in sets])]
        close = frm[BaseExtra.closestvector(target, frm)]
        far = frm[BaseExtra.farthestvector(target, frm)]
        b = np.linalg.norm(np.array(far-close))/(np.linalg.norm(np.array(start-close))+np.linalg.norm(np.array(start-far)))
        start = np.array(BaseExtra.multimove(start, close, b))
    return start