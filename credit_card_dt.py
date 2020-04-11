import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import tree

numerical_labels = ['ID', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
                    'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
                    'CNT_FAM_MEMBERS', 'MONTHS_BALANCE', 'STATUS']
categorical_labels = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

df = pd.read_csv('./data/final_status_preserve.csv',header=0, encoding='utf-8')
for all_cat in list(df):
    df[all_cat].replace('', np.nan, inplace=True)
    df.dropna(subset=[all_cat], inplace=True)

encoder = LabelEncoder()
for cat in categorical_labels:
    labels = encoder.fit_transform(df[cat])
    df[cat] = labels

# input variables
x = df.values[:, 1:17]
# output variables
y = df.values[:, 17:18]

# 80/20 split between training and validation set. Can try 90/10 split since there is little data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

verbose = True
all_depth = []
for depth in range(1, 16):
    max_accuracy = -1
    param = (0, 0, 0, 0, 0)
    for leaf in range(2, 16):
        for impurity in range(1, 16):
            for samples in range(1, 16):
                clf = tree.DecisionTreeClassifier(
                    max_depth=depth,
                    max_leaf_nodes=leaf,
                    min_impurity_decrease= impurity,
                    min_samples_leaf=samples
                )
                clf = clf.fit(x_train, y_train)
                y_predict = clf.predict(x_test)
                accuracy = accuracy_score(y_test, y_predict)
                if verbose:
                    print('Depth: %d, Leaf: %d, Impurity: %d, Samples: %d, Accuracy: %.3f' % 
                        (depth, leaf, impurity, samples, accuracy))   
                if accuracy > max_accuracy:
                    param = (depth, leaf, impurity, samples, accuracy)
    if verbose:
        print('At depth: %d, best params: %d %d %d' % (depth, param[1], param[2], param[3]))
    all_depth.append(param)

print('--------------FINAL RESULTS-----------------')
for p in all_depth:
    print('Depth: %d, Leaf: %d, Impurity: %d, Samples: %d, Accuracy: %.3f' % p)   
print('--------------END FINAL RESULTS--------------')


# --------------FINAL RESULTS-----------------
# Depth: 1, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 2, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 3, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 4, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 5, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 6, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 7, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 8, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 9, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 10, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 11, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 12, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 13, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 14, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# Depth: 15, Leaf: 15, Impurity: 15, Samples: 15, Accuracy: 0.873
# --------------END FINAL RESULTS--------------
    