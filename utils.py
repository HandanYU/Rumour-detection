import time

def timer(unit):
    def timer_deco(f):
        def fn(*args, **kw):
            t1 = time.time()
            r = f(*args, **kw)
            t2 = time.time()
            t = t2 - t1 if unit == 's' else (t2 - t1) * 1000
            print('Call {} in {} {}'.format(f.__name__, t, unit))
            return r
        return fn
    return timer_deco