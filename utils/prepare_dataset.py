import pandas as pd
from sklearn.utils import shuffle
import text_util

pos1 = text_util.load_txt("..\raw_data\books\review_negative")
pos2 = text_util.load_txt("..\raw_data\books\review_positive")
file_object=[]
file_object=pos1+pos2

try:
     all_the_text = file_object.read( )
finally:
     file_object.close( )
r = all_the_text.split('\n')
r=list(r)
print(r)
result =pd.DataFrame(columns={'comment','label'})
for i in range(len(r)):
        if i<3000:
           result=result.append({'comment':r[i],'label':1},ignore_index=True)
        else:
           result = result.append({'comment': r[i], 'label': 0}, ignore_index=True)

result=shuffle(result)
#print(result)
result.to_csv("..\data\books.csv",encoding='utf-8')
