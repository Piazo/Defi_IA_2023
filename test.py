import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_moons
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
plt.scatter(X[:,0], X[:,1], c=y)

xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)


# model = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=100)

# model.fit(xtr, ytr)
# print(model.score(xte, yte))

model = RandomForestClassifier(n_estimators=100, max_depth=100)

model.fit(xtr, ytr)
print(model.score(xte, yte))


model = AdaBoostClassifier(n_estimators=100)

model.fit(xtr, ytr)
print(model.score(xte, yte))




mod1 = SGDClassifier(random_state=0)
mod2 = DecisionTreeClassifier(random_state=0)
mod3 = KNeighborsClassifier(n_neighbors=2)

mod4 = StackingClassifier([ ('SGD', mod1),
                            ('Tree', mod2),
                            ('KNN', mod3)],
                            final_estimator=KNeighborsClassifier())

print(mod4.fit(xtr, ytr).score(xte, yte))
# mod4 = VotingClassifier([   ('SGD', mod1),
#                             ('Tree', mod2),
#                             ('KNN', mod3)],
#                             voting='hard')

# for mod in (mod1, mod2, mod3, mod4):
#     mod.fit(X_train, y_train)
#     print(mod.__class__.__name__, mod.score(X_test, y_test))










"""
---------------- Note pour Antonio ----------------

Utiliser des ensembles de modeles

On va utiliser du BAGGING lorsque qu'on a plusieurs modeles qui vont avoir tendence a faire de l'OVERFITTING

On va utiliser du BOOSTING lorsque qu'on a plusieurs modeles qui vont avoir tendence a faire de l'UNDERFITTING

On va utiliser du STACKING lorsque qu'on a plusieurs modeles que l'on va avoir entrainé avec beaucoup de données





"""