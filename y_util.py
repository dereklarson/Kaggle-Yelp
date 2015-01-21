import time, json

def timet(func, **args):
    start = time.clock()
    ret = func(**args)
    print 'Took', time.clock() - start, 'seconds'
    return ret

def read_json(fname):
    with open(fname) as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    return data

def age_in_hours(date, ref):
    conv = time.strptime(date, '%Y-%m-%d')
    y2k = time.strptime(ref, '%Y-%m-%d')
    return abs(time.mktime(conv) - time.mktime(y2k))/ 3600.
