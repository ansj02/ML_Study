import pandas_datareader.data as web
import datetime

start = datetime.datetime(2018, 12, 19)
end = datetime.datetime(2020, 3, 1)

gs = web.DataReader("035720.KS", "yahoo", start, end)

Change_Value=[]

for i in range(len(gs['Close'])) :
    Change_Value.append(gs['Close'][i] - gs['Close'][i-1])

while(len(Change_Value)%10 != 0) :
    del Change_Value[0]

ComparisonTemp=[]
inputdataTemp=[]
n=10
for i in range(len(Change_Value)) :
    if i%n == n-1 :
        ComparisonTemp.append(Change_Value[i])
    else :
        inputdataTemp.append(Change_Value[i])
n=9
inputdata = [inputdataTemp[i * n:(i + 1) * n] for i in range((len(inputdataTemp) + n - 1) // n )]
#Comparison = [ComparisonTemp[i * 1:(i + 1) * 1] for i in range((len(ComparisonTemp)) // 1 )]
Comparison = ComparisonTemp
print(len(Comparison))
print(len(inputdata))
print(Change_Value)
print(inputdata)
'''
money1 = 10000
money2 = 10000
money3 = 10000


hav = 0
for tVal in Change_Value :
    if hav == 1 : money1 += tVal

    if tVal > 0 : hav = 1
    else : hav = 0
hav = 0
for tVal in Change_Value :
    if hav == 1 : money2 += tVal

    if tVal < 0 : hav = 1
    else : hav = 0

hav = 0
for tVal in Change_Value :
    if hav == 1 : money3 += tVal

    hav = 1

print(money1)
print(money2)
print(money3)
'''