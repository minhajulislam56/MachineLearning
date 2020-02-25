import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = [0,1,2,3]
y = [0,2,4,6]
xlab = ['A', 'B', 'C', 'D']

bars = plt.bar(xlab, x)
# bars[1].set_hatch('/')
patterns = ['%', '/', '0', '*']
i = 0
for bar in bars:
    # bar.set_hatch(patterns[i])
    bar.set_hatch(patterns.pop(0))
    i = i+1
plt.show()

# Resizing Graph
# plt.figure(figsize=(5,3), dpi=100)

a = np.arange(0, 5, 0.5)    # select interval in plot
# plot with argument
plt.plot(a[:6], a[:6]**2, label='2x', color='blue', ls='-', linewidth=2, marker='.', ms=10, mfc='red')

plt.plot(a[5:], a[5:]**2, 'b--', label='approximate', linewidth=2)
# Shorthand notation [color][marker][line]
#plt.plot(x,y, 'b-.', label='2x')

plt.legend()
# Title (specify font with fontdict)
plt.title('New Graph!', loc='right', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([0,1,2,3,4,5])
plt.yticks([0,2,4,6,8,10])
plt.show()

# plt.savefig('first_graph.png', dpi=300)