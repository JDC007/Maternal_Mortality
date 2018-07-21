import pandas as pd
from sklearn.preprocessing import Imputer
import numpy


df=pd.read_csv("maternal_mortality.csv")
fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

df.fillna(df.mean()).to_csv('with_mean.csv')

'''
feature_column_names= ['qtype','qweight','qtype2','q201', 
               'q213_01',  'q308', 'q313_01',  'q325',  'q508_1',
            'q509a_1'
            , 'q509b_1'
            , 'q509c_1'
            , 'q509d_1'
            , 'q510_1'
            ,'q512_1'
            , 'q513_1'
            , 'q518_1'
            
            , 'q527_1'
            , 'q528_1',
            'q533', 'q534', 'q701', 'q702', 'q703', 'q704',
             'q705', 'q706']

x= df[feature_column_names].values
x = fill_0.fit_transform(x)


#new_f=fill_0.fit_transform(f)
#numpy.savetxt('test.csv',new_f ,delimiter=',')

pd.DataFrame(x).to_csv('prediction.csv')
'''
