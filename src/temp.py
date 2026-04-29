import numpy as np

sum = 0
for n in range(-1, 2):
    for m in range(-1, 2):
        if n != m:
            sum += (n-m)**2
            print(n, m, (n-m)**2)
print(sum)

sum2 = 0
sum3 = 0
for k in range(-1, 2):
    if k != 0:
        sum2 += k**2
for m in range(-1, 2):
    sum3 += 1
print(sum2*sum3)
