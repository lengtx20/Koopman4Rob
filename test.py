from collections import deque

d = deque(maxlen=None)
d.append(1)  # 右端入队
d.appendleft(2)  # 左端入队
print(d)  # deque([2, 1])
d.pop()  # 右端出队 → 1
d.popleft()  # 左端出队 → 2
