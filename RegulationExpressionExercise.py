import re

exampleString = '''
Jass is 15 years old, and Daniel is 27 years old.
Edward is 97, and his grandfather, Oscar, is 102.
'''

ages = re.findall(r'\d{1,3}',exampleString)

# we are looking for one Capital and more lower letter
names = re.findall(r'[A-Z][a-z]*',exampleString)

print(ages)
print(names)

ageDict = {}
x=0
for eachname in names:
    ageDict[eachname] = ages[x]
    x+=1
print(ageDict)
