import pandas as pd
import numpy as np
import features


pricing_requests = []
# print(np.load('./data/requestHistory.npy')[1])
r1 = np.load('./data/responseHistory.npy',  allow_pickle=True)[1]

requests = [r1]
for r in requests:
    pricing_requests.append(
        pd.DataFrame(r.json()['prices']).assign(**r.json()['request'])
    )
pricing_requests = pd.concat(pricing_requests)
print(pricing_requests)