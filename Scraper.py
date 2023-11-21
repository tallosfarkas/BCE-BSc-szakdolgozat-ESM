import eikon as ek
import numpy as np
import arch
import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model

# Set your app key
ek.set_app_key('37934fabc6604c9bb80b65b1e3fa6bd265ad228b')


# Define the option symbol and date range
option_symbol = 'DAIGnOPTTOT.EX'
start_date = '2021-01-01'
end_date = '2021-12-31'

# Fetch the volume data
volume_data = ek.get_timeseries(option_symbol, fields=['VO'], start_date=start_date, end_date=end_date)

# Plotting the volume data
volume_data.plot(y='VO', figsize=(10,6))
plt.title('Volume Data for DAIGnOPTTOT.EX')
plt.ylabel('Volume')
plt.xlabel('Date')
plt.show()


import eikon as ek



# Specify the symbol of the option and the date range
option_symbol = 'DAIGnOPTTOT.EX'
start_date = '2020-01-01'
end_date = '2022-12-31'

# Fetch the volume data
volume_data, err = ek.get_data(option_symbol, ['TR.VOLUME'], start_date=start_date, end_date=end_date)

# Now, volume_data should contain the historical volume data for the specified option


ek.get_timeseries(['DAIGnOPTTOT.EX'], ['VOLUME'])




import refinitiv.data as rd
from refinitiv.data.discovery import Chain
 
rd.open_session()

aapl = Chain(name="0#AAPL*.U")
print(len(aapl.constituents))
chain = Chain(name="0#MBGn*.DE")

 
# print number of chain's constituents
print(len(chain.constituents))
 
# get history of first 10 constituents in the chain (for example)
rd.get_history(chain.constituents[:10])

efom_chains = list(rd.discovery.Chain('DAIGn:'))
efom_chains