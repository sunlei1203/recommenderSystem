'''Description: this is a simple recommendation sample'''
from math import sqrt
critics = {'Lisa rose':{'Lady in the water':2.5,'snakes on a plane':3.5,'Just My Luck':3.0,'superman Returns':3.5,'You,Me and Dupree':2.5,'The Night Listener':3.0},'Gene Seymour': {'Lady in the water':3.0,'snakes on a plane':3.5,'Just My Luck':3.5,'superman Returns':5.0,'You,Me and Dupree':3.0,'The Night Listener':3.5},'Michael Phillipse':{'Lady in the water':2.5,'snakes on a plane':3.0,'superman Returns':3.5,'The Night Listener':4.0},'Claudia Puig':{'snakes on a plane':3.5,'Just My Luck':3.0,'superman Returns':4.0,'You,Me and Dupree':2.5,'The Night Listener':4.5},'Mick LaSalle':{'Lady in the water':3.0,'snakes on a plane':4.0,'Just My Luck':2.0,'superman Returns':4.0,'You,Me and Dupree':2.0},'Jack Matthews':{'Lady in the water':3.0,'snakes on a plane':4.0,'Just My Luck':2.0,'superman Returns':3.0,'You,Me and Dupree':3.0,'The Night Listener':2.0},'Toby':{'snakes on a plane':4.5,'superman Returns':4.0,'You,Me and Dupree':1.0}}
print critics['Lisa rose']['Lady in the water']
print critics['Toby']
def sim_pearson(prefs,p1,p2):
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]:si[item]=1
    n=len(si)
    if n==0:return 1
    sum1 = sum([prefs[p1][it] for it in si]) 
    sum2 = sum([prefs[p2][it] for it in si]) 
    sum1Sq = sum([pow(prefs[p1][it],2) for it in si]) 
    sum2Sq = sum([pow(prefs[p2][it],2) for it in si]) 
    pSum = sum([prefs[p1][it]*prefs[p2][it] for it in si])
    num = pSum-(sum1*sum2/n)
    den = sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den ==0:return 0
    return num/den
def topMatches(prefs,person,n=5,similarity=sim_pearson):
    scores = [(similarity(prefs,person,other),other)for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]
def getRecommendations(prefs,person,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in prefs:
        if other == person:continue
        sim = similarity(prefs,person,other)
        if sim<=0:continue
        for item in prefs[other]:
            if item not in prefs[person] or prefs[person][item]==0:
                totals.setdefault(item,0)
                totals[item]+=sim*prefs[other][item]
                simSums.setdefault(item,0)
                simSums[item]+=sim
    rankings = [(total/simSums[item],item) for item,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    return rankings
print sim_pearson(critics,'Lisa rose','Gene Seymour')
print topMatches(critics,'Toby',n=3)
print getRecommendations(critics,'Toby')