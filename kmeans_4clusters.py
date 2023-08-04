import pandas as pd

#Carregando Dados do Enade
df = pd.read_csv("EstudSMgerados_2022_08_13.csv") #print (df)
del df['index']

from datetime import datetime
dh1 = datetime.today()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(df)

dh2 = datetime.today()
tempo_gasto =  dh2 - dh1
print ('tempo_gasto:',tempo_gasto)

#df.rename(columns={0: 'softwareUnderstanding'}, inplace = True)
#df.rename(columns={1: 'understandingConcepts'}, inplace = True)
#df.rename(columns={2: 'practiceSM'}, inplace = True)
#df.rename(columns={3: 'testSM'}, inplace = True)

df['k-classes'] = kmeans.labels_

#import seaborn as sb
#sb.pairplot(df, hue='k-classes')

df