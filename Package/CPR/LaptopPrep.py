import DataPrepBase as dbase

def ghzConvert(ghz):
    newwordlist = list()
    for i in ghz.replace('GHz', '')[::-1]:
        if i == ' ':
            return float(''.join(newwordlist[::-1]))
        newwordlist.append(i)

def rezConvert(rez):
    newwordlist = list()
    switch = True
    for i in rez[::-1]:
        if switch == True:
            if i == ' ':
                switch = False
            newwordlist.append(i)
    new = list(''.join(newwordlist[::-1]).split('x'))
    return float(new[0])*float(new[1])

def memConvert(mem):
    newwordlist = list()
    switch = True
    for i in mem:
        if switch == True:
            if i == ' ':
                switch = False
            newwordlist.append(i)
    new = ''.join(newwordlist)
    if 'GB' in new:
        return float(new.replace('GB', ''))
    if 'TB' in new:
        return float(new.replace('TB', ''))*1024

def full(dataset):
    dft = dataset.copy()
    dft.Inches = [float(t) for t in dft.Inches]
    dft.ScreenResolution = [rezConvert(t) for t in dft.ScreenResolution]
    dft.Cpu = [ghzConvert(t) for t in dft.Cpu]
    dft.Ram = [float(t[:-2]) for t in dft.Ram]
    dft.Memory = [memConvert(t) for t in dft.Memory]
    dft.Weight = [float(t[:-2]) for t in dft.Weight]
    dft.Price_euros = [float(t) for t in dft.Price_euros]
    return dft

def numerical(dataset):
    dft = dataset.copy()
    dft.Inches = [float(t) for t in dft.Inches]
    dft.ScreenResolution = [rezConvert(t) for t in dft.ScreenResolution]
    dft.Cpu = [ghzConvert(t) for t in dft.Cpu]
    dft.Ram = [float(t[:-2]) for t in dft.Ram]
    dft.Memory = [memConvert(t) for t in dft.Memory]
    dft.Weight = [float(t[:-2]) for t in dft.Weight]
    dft.Price_euros = [float(t) for t in dft.Price_euros]
    return dft[['Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Weight', 'Price_euros']].drop_duplicates()