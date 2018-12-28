import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(10000) + 5
y = np.random.exponential(5, 100000)
bc, be = np.histogram(x, bins=np.linspace(0, 15, 26))
byc, _ = np.histogram(y, bins=np.linspace(0, 15, 26))
bm = (be[:-1] + be[1:])/2
bw = be[1:] - be[:-1]

# plt.bar(bm, byc, width=bw)


plt.hist([bm, bm], weights=[byc, bc], bins=be, stacked=True, edgecolor='black', lw= 0.5, histtype="stepfilled")

# plt.plot(bm, byc, drawstyle='steps-mid', color='black', lw=2)
# plt.bar(bm, bc, width=bw, bottom=byc)
# plt.plot(bm, byc+bc, drawstyle='steps-mid', color='black', lw=2)
# plt.savefig('test.png')
plt.show()