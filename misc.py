import time

class Timer:
    def __enter__(self):
        self.dt = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dt = time.time() - self.dt


# def foo():
#     with Timer() as t:
#         t0 = t.dt
#         time.sleep(1)
#         t1 = t.dt
#         print(t1 - t0)