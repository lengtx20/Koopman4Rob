import timeit
from functools import partial


def func(a, b, c):
    return a + b + c


# 方式1: 直接调用
def direct():
    return func(1, 2, 3)


# 方式2: partial 全绑定
pfunc = partial(func, 1)


def via_partial():
    return pfunc(2, 3)


# 方式3: 字典 + **解包
args_dict = {"a": 1, "b": 2, "c": 3}


def via_unpack():
    return func(**args_dict)


# 测试
number = 500_000
t1 = timeit.timeit(direct, number=number)
t2 = timeit.timeit(via_partial, number=number)
t3 = timeit.timeit(via_unpack, number=number)

print(f"直接调用:     {t1:.6f}s")
print(f"partial:      {t2:.6f}s  ({t2 / t1:.2f}x)")
print(f"**kwargs:     {t3:.6f}s  ({t3 / t1:.2f}x)")
