# I have removed away the column on DAYS_BIRTH and DAYS_EMPLOYED
# The values made no sense to this tiny brain so i casually left it out.
# If anyone has any idea on how to interpret and convert the data, pls
# feel free to do so and include it as one of our input variables

import pandas as pd

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split

numerical_labels = ['ID', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
                    'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
                    'CNT_FAM_MEMBERS', 'MONTHS_BALANCE', 'STATUS']
categorical_labels = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

df = pd.read_csv('./data/legit_clean.csv',header=0, encoding='utf-8')

# One hot encode all categorical data labels. Labels are changed to ORIGINAL_LABEL_SUBLABEL_NAME
df = pd.get_dummies(df, columns=categorical_labels, prefix=categorical_labels)
# Drop status column which is the target output column
df_drop_status = df.drop('STATUS', axis=1)

# Get all input variables
x = df_drop_status.values[:, 1:(len(df_drop_status.columns) - 1)]

# status column
y = df.values[:, 17:18]

# 80/20 split between training and validation set. Can try 90/10 split since there is little data
# TODO SMOTE oversampling possible?
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
# he_uniform is a truncated normal distribution for random initial weights of neurons
model.add(Dense(110, input_dim=len(x[0]), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# TODO K-Fold Validation to evaluate the model?
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=10, verbose=1)

_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# This extremely basic model achiveves 83% accuracy rate on train data and 82% on test data


# Other things to consider doing. 
# TODO evaluating our dataset. How much is missing, how much is useful. Information value can give some insight into this
# TODO try out other models like decision trees
# TODO try NOT one hot encoding the categorical variables. have some other form of representation for these types of variables
# TODO plot something? with pyplot. maybe test accuracy etc...