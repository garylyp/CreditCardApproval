# Joins up the months balance and status into main table. 
# Months is taken to be the first month the data started tracking a certain user ID
# status is converted to a binary variable. Anything X,C,0,1 is counted as
# 0 and >= 2 is counted as 1
# 0 means good credit record and the bank should probably approve this applicant
# 1 means bad credit record and the bank should probably deny this applicant

import pandas as pd

from numpy import array

df = pd.read_csv('./data/cleaned_application_record_sorted.csv',header=0, encoding='utf-8')
record = pd.read_csv('./data/cleaned_credit_record_sorted.csv', header=0, encoding='utf-8')

replaced = record.replace({'STATUS' : {
    'X' : 0,
    'C' : 0,
    '0' : 0,
    '1' : 0,
    '2' : 1,
    '3' : 1,
    '4' : 1,
    '5' : 1}})
record_group_id = replaced.groupby(['ID'])
status_targets = pd.DataFrame(record_group_id['STATUS'].agg(max))
begin_month=pd.DataFrame(record_group_id['MONTHS_BALANCE'].agg(min))
# default join strategy is inner join. SQL-like intersection join
new_data=pd.merge(df,begin_month,on="ID")
new_data=pd.merge(new_data, status_targets,on='ID')

new_data.to_csv('./data/final_clean_version.csv', index=False)
