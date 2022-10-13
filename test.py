import features
import pandas as pd
import numpy as np

# Création d'un tableau de requêtes : 

requested = features.getAllRequests()
print(len(requested))
requested = requested[0:len(requested)-1]
print(len(requested), requested)