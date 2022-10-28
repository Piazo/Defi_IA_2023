import features
import re

req = features.getAllRequests()
resp = features.getAllResponses()

for i in range(len(req)):
    if "nameAvatar" in req[i]:
        try:
            print(int(re.findall(r"(\d+)[,}]", resp[i].text.split('"id":',1)[1])[0]))
            print(resp[i].text.split('"name":"',1)[1].split('"',1)[0])
        except:
            pass

"""
listID = []
listName = []
for i in range(len(req)):
    if "nameAvatar" in req[i]:

        # Alors c'est le bordel mais en gros ca recupere le texte apres "id", 
        # puis on garde juste les chiffres avant un autre caractere
        # grace a cette belle expression reguliere  et ca donne l'ID c'est fabuleux
        # Puis on met un indice 0 a la fin psk cest une liste et qu'on veut que le 1er elem
        # et on le cast en integer pour avoir le bon format et GG
        listID.append(int(re.findall(r"(\d+)[,}]", resp[i].text.split('"id":',1)[1])[0]))
        # Alors la en gros on fait pareil mais on split deux fois et hopla magie
        # On recupere le nom
        listName.append(resp[i].text.split('"name":"',1)[1].split('"',1)[0])
"""