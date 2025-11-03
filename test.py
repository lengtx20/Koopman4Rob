import timeit

n = 100_000


# 方法1：预分配 + 索引赋值
def prealloc_index():
    lst = [None] * n
    for i in range(n):
        lst[i] = i
    return lst


# 方法2：动态 append
def dynamic_append():
    lst = []
    for i in range(n):
        lst.append(i)
    return lst


n = 100_000

time1 = timeit.timeit(prealloc_index, number=100)
time2 = timeit.timeit(dynamic_append, number=100)

print(f"Prealloc + index: {time1:.3f}s")
print(f"Dynamic append:   {time2:.3f}s")
print(f"Improvement:      {time2 / time1:.2f}x faster")
