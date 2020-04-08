# Script sorts all values(according to ID) and outputs them to another file
# There is not much reason to do this. It was just very irritating to see 
# the ID not being in sequential order

import pandas as pd
from numpy import array

main_df = pd.read_csv('./data/application_record_no_birth_employ_date.csv', 
                    dtype=str,
                    header=0)

output_df = pd.read_csv('./data/credit_record.csv',
                        dtype=str,
                        header=0)

sorted_main_df = main_df.sort_values(by=['ID'])
sorted_output_df = output_df.sort_values(by=['ID', 'MONTHS_BALANCE'])

sorted_main_df.to_csv('./data/cleaned_application_record_sorted.csv', index=False)
sorted_output_df.to_csv('./data/cleaned_credit_record_sorted.csv', index=False)