import features
import pandas as pd
import numpy as np

# Création d'un tableau de requêtes : 

tab_request = [
                ["Avataricard01", 'austrian', 'amsterdam', 44, 1],
                ["Avataricard01", 'austrian', 'copenhagen', 44, 0],

                ["Avataricard01", 'bulgarian', 'amsterdam', 44, 1],
                ["Avataricard01", 'bulgarian', 'copenhagen', 44, 0],

                ["Avataricard01", 'croatian', 'madrid', 44, 1],
                ["Avataricard01", 'croatian', 'paris', 44, 0],

                ["Avataricard01", 'cypriot', 'madrid', 44, 0],
                ["Avataricard01", 'cypriot', 'paris', 44, 1],

                ["Avataricard01", 'danish', 'rome', 20, 1],
                ["Avataricard01", 'danish', 'sofia', 20, 0],

                ["Avataricard01", 'dutch', 'rome', 20, 0],
                ["Avataricard01", 'dutch', 'sofia', 20, 1],

                ["Avataricard01", 'estonian', 'valletta', 20, 0],
                ["Avataricard01", 'estonian', 'vienna', 20, 1],

                ["Avataricard01", 'finnish', 'valletta', 20, 0],
                ["Avataricard01", 'finnish', 'vienna', 20, 1],

                ["Avataricard01", 'german', 'vilnius', 20, 0],
                ["Avataricard01", 'german', 'sofia', 20, 1],

                ["Avataricard01", 'irish', 'amsterdam', 20, 0],
                ["Avataricard01", 'irish', 'copenhagen', 20, 1],

                ["Avataricard01", 'italian', 'madrid', 20, 1],
                ["Avataricard01", 'italian', 'paris', 20, 0],

                ["Avataricard01", 'latvian', 'rome', 20, 0],
                ["Avataricard01", 'latvian', 'paris', 20, 1],

                ["Avataricard01", 'lithuanian', 'valletta', 20, 0],
                ["Avataricard01", 'lithuanian', 'vienna', 20, 1],

                ["Avataricard01", 'luxembourgish', 'vilnius', 20, 0],
                ["Avataricard01", 'luxembourgish', 'sofia', 20, 1],

                ["Avataricard01", 'maltese', 'amsterdam', 12, 0],
                ["Avataricard01", 'maltese', 'copenhagen', 12, 1],

                ["Avataricard01", 'polish', 'madrid', 12, 0],
                ["Avataricard01", 'polish', 'paris', 12, 1],

                ["Avataricard01", 'portuguese', 'rome', 12, 0],
                ["Avataricard01", 'portuguese', 'valletta', 12, 1],

                ["Avataricard01", 'romanian', 'vienna', 12, 0],
                ["Avataricard01", 'romanian', 'vilnius', 12, 1],

                ["Avataricard01", 'slovakian', 'amsterdam', 12, 0],
                ["Avataricard01", 'slovakian', 'copenhagen', 12, 1],

                ["Avataricard01", 'slovene', 'madrid', 12, 0],
                ["Avataricard01", 'slovene', 'paris', 12, 1],

                ["Avataricard01", 'spanish', 'rome', 12, 0],
                ["Avataricard01", 'spanish', 'sofia', 12, 1],
            ]