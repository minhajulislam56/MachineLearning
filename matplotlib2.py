import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gas = pd.read_csv('gas_prices.csv')

for country in gas:
    if country != 'Year':
        plt.plot(gas.Year, gas[country], marker='.', ls='-', label=country)

plt.title('Gas Price over Time in ($)')
# plt.plot(gas.Year, gas.USA, 'b.-', label='USA')
# plt.plot(gas.Year, gas.Canada, 'r.-', label='Canada')
# plt.plot(gas.Year, gas['South Korea'], 'y.-', label='South Korea')
plt.legend()
plt.xticks(gas.Year[::3])
plt.xticks(gas.Year[::3].tolist()+[2011])
plt.show()