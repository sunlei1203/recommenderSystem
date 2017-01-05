from math import sqrt
def loadMovieLens(path='MyProject/ml-100k'):
    movies = {}
    for line in open(path+'/u.item'):
        (id,title) = line.split('|')[0:2]
        movies[id]=title    
    #print movies
    prefs={}
    for line in open(path+'/u.data'):
        (user,movieid,rating,ts) = line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movies[movieid]] = float(rating)
    return prefs
critics = loadMovieLens()
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
# print critics['87']
print getRecommendations(critics,'87')[0:30]