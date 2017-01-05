import time
import random
import math

people = [('Seymour','BOS'),('Franny','DAL'),('Zooey','CAK'),('Walt','MIA'),('Buddy','ORD'),('Les','OMA')]
destination = 'LGA'
flights = {}
for line in file('MyProject/dataSet/schedule.txt'):
    origin,dest,depart,arrive,price = line.strip().split(',')
    flights.setdefault((origin,dest),[])
    flights[(origin,dest)].append((depart,arrive,int(price)))
# print str(flights)
def getminutes(t):
    x = time.strptime(t,'%H:%M')
    return x[3]*60+x[4]
def printschedule(r):
    for p in range(len(people)):
#         print 'p: '+str(p)+'\t'
        name = people[p][0]
#         print 'name = people['+str(p)+'][0]: '+str(name)+'\t'
        origin = people[p][1]
#         print 'origin = people['+str(p)+'][1]: '+str(origin)+'\t'
        start = flights[(origin,destination)][r[2*p]]
#         print 'start: '+str(start)+'\t'
#         print 'destination: '+str(destination)+'   '+str(flights[(origin,destination)])
        ret = flights[(destination,origin)][r[2*p+1]]
#         print 'ret: '+str(ret)+'\n'
        print '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' %(name,origin,start[0],start[1],start[2],ret[0],ret[1],ret[2])
# print str(people)
# print str(people[0])
# print str(people[0][0])
r = [1,4,3,2,7,3,6,3,2,4,5,3]
printschedule(r)

def schedulecost(sol):
    totalprice = 0
    latestarrival =0
    earliestdep = 24*60
    for d in range(len(sol)/2):
        origin = people[d][1]
        depart = flights[(origin,destination)][int(sol[2*d])]
        return1 = flights[(destination,origin)][int(sol[2*d+1])]
        totalprice += depart[2]
        totalprice += return1[2]
        if latestarrival<getminutes(depart[1]):latestarrival = getminutes(depart[1])
        if earliestdep>getminutes(return1[0]): earliestdep = getminutes(return1[0])
    totalwait = 0
    for p in range(len(sol)/2):
        origin = people[p][1]
        depart = flights[(origin,destination)][int(sol[2*p])]
        return1 = flights[(destination,origin)][int(sol[2*d+1])]
        wait = latestarrival-getminutes(depart[1])
        totalwait += wait
        totalwait += getminutes(return1[0])-earliestdep
    if latestarrival>earliestdep:totalprice += 50
    return totalprice+totalwait
print str(schedulecost(r))
def randomoptimize(domain,costf):
    best = 999999999
    bestr = None
    for i in range(1000):
        r = [random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        #print str(r)
        cost = costf(r)
        if cost<best:
            best = cost
            best = r
    return r
def hillclimb(domain,costf):
    sol = [random.randint(domain[i][0],domain[i][1])for i in range(len(domain))]
    print str(sol)+'  '+str(domain[0][0])+' '+str(domain[0][1])+' '+str(len(domain))
    while 1:
        neighbors = []
        for j in range(len(domain)):
            if sol[j]>domain[j][0]:
                neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])
            if sol[j]<domain[j][1]:
                neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
        current = costf(sol)
        best = current 
        for j in range(len(neighbors)):
            cost= costf(neighbors[j])
            if cost<best:best = cost
            sol = neighbors[j]
            if best == current:break
    return sol
domain = [(0,9)*(len(people*2))]
#s = hillclimb(domain,schedulecost)
s = randomoptimize(domain,schedulecost)
schedulecost(s)