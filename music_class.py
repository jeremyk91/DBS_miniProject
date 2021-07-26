#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:51:11 2021

@author: jeremykuek
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


features_df = pd.read_csv("features.csv")
features_df.drop_duplicates(inplace=True)

labels_df = pd.read_csv("labels.csv")
test_df = pd.read_csv("test.csv")


#%%

features_df = features_df.merge(labels_df, on = 'trackID')
X_train = features_df[features_df.columns[:-1]]
y_train = features_df[features_df.columns[-1]]

#%%
# features_df['genre'].value_counts()

"""
classic pop and rock     1684
folk                     1665
metal                    1209
soul and reggae           988
punk                      981
pop                       731
dance and electronica     523
jazz and blues            347
Name: genre, dtype: int64

"""
#%%
# features_df.dropna(inplace=True)

# features_df = features_df.fillna(features_df.mean())
#%%
SnR = features_df[features_df['genre']=='soul and reggae']

#%%

genre_list = ['classic pop and rock', 'folk', 'metal','soul and reggae','punk','pop','dance and electronica','jazz and blues']            

#%%

def ecdf(data):
    x_axis = np.sort(data)
    y_axis = np.arange(1,(len(data)+1))/len(data)
    return x_axis,y_axis

#%%


# for genre in genre_list:
#     features_df[features_df['genre']==genre]['loudness'].plot(label = genre)
#     plt.legend()
#     plt.show()

#%%
feature_list = ['loudness','tempo','time_signature','key','mode','duration'] # vect_1 ... vect_148

feature = 'duration'

## mode is the only categorical

for genre in genre_list:
    df = features_df[features_df['genre']==genre][feature]
    x,y = ecdf(df)
    plt.plot(x,y,linestyle='none',marker='.',label=genre)
plt.legend()
plt.title(f"{feature} ECDF")
plt.margins(0.02)
plt.show()

## there is an issue where for many features, there's quite a lot of outliers..hence its better to use standardScaler




#%%
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, quantile_transform
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

#%%

def get_models():
	models, names = list(), list()
	# LR
	models.append(LogisticRegression(max_iter=10000, tol=0.01))
	names.append('LR')
	# LDA
	models.append(LinearDiscriminantAnalysis())
	names.append('LDA')
	# SVM
	models.append(LinearSVC())
	names.append('SVM')
	# KNN
	models.append(KNeighborsClassifier(n_neighbors=5))
	names.append('KNN_5')
    
    # models.append(KNeighborsClassifier(n_neighbors=7))
    # names.append('KNN_7')
	return models, names

models, names = get_models()

#%%
vect_list = ['vect_'+str(i) for i in range(1,148+1)]
cont_features = ['loudness','tempo','time_signature','duration']
cont_features.extend(vect_list)

cat_features = ['key','mode']



# scaler = MinMaxScaler()
# vect_df = features_df[cont_features].copy()
# vect_df = scaler.fit_transform(vect_df)
# print(vect_df)

# pca = PCA(n_components=0.95) # retain 95% variance
# pca.fit(vect_df)
# print(f"n_components_: {pca.n_components_}")
# print(f"explained_variance_: {pca.explained_variance_}")

"""
n_components_: 42
explained_variance_: 
[0.40966774 0.235753   0.09402189 0.05365637 0.04686117 0.03384986
 0.02984307 0.02625552 0.02585401 0.02545205 0.02131297 0.02030012
 0.01708813 0.0161066  0.01465899 0.01392674 0.0123173  0.01139543
 0.01063066 0.00975763 0.00922083 0.00848699 0.00802696 0.00739309
 0.00699523 0.00652382 0.00620832 0.00610444 0.00576026 0.00511901
 0.00493193 0.00466155 0.00425483 0.00415285 0.00392898 0.0038707
 0.00370219 0.00352066 0.00342531 0.00328836 0.0032047  0.00287089]

"""


#%% Balance out the training data

# y_train = pd.DataFrame(y_train)


# for genre in y_train['genre'].unique():
#     print(genre)

# y_train.value_counts().min()

#%%

# numeric_features = ['age', 'fare']
for model,name in zip(models,names):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), # this will fillna() with mean/median for all missing values in each column.
                                          ('scaler',StandardScaler() ) # MinMaxScaler
                                          ])
    
    # categorical_features = ['embarked', 'sex', 'pclass']
    # categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    
    # qt = quantile_transform(X_train[cont_features])
    
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, cont_features),
                                                    # ('qt', qt)
                                                    # ('cat', categorical_transformer, categorical_features)
                                                    ])
    
    pca = PCA(n_components=0.95)
    
    # model = LogisticRegression(max_iter=10000, tol=0.01)
    # model = LinearSVC()
    # model = GaussianNB()
    
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('pca', pca),
                          ('classifier', model) #  GaussianNB()
                          ])  
    
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_train)
    print(f"{name} classification report:\n{classification_report(y_train, y_pred)}")


# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
# n_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report pipeline performance
# print(f"Accuracy: {round(np.mean(n_scores),3)} +/- {round(np.std(n_scores), 3)}")


"""
We have a situation whereby classes with more support have higher F1 score. TO combat the imbalanced dataset,
try stratifiedKfold 

"""

#%%


# # ensure inputs are floats and output is an integer label
# X = X.astype('float32')
# y = LabelEncoder().fit_transform(y.astype('str'))
# # define the pipeline
# trans = MinMaxScaler()
# model = KNeighborsClassifier()
# pipeline = Pipeline(steps=[('t', trans), ('m', model)])
# # evaluate the pipeline
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# # report pipeline performance
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


#%%
"""
Probably have to design the model to use 100% cont. variables or 100% cat. variables.

"""

#%%
"""
========
Features
========

* trackID: unique identifier for each song (Maps features to their labels)
* title: title of the song. Type: text.
* tags: A comma-separated list of tags representing the words that appeared in the lyrics of the song and are assigned by human annotators. Type: text / categorical.
* loudness: overall loudness in dB. Type: float / continuous.
* tempo: estimated tempo in beats per minute (BPM). Type: float / continuous.
* time_signature: estimated number of beats per bar. Type: integer.
* key: key the track is in. Type: integer/ nominal. 
* mode: major or minor. Type: integer / binary.
* duration: duration of the song in seconds. Type: float / continuous.
* vect_1 ... vect_148: 148 columns containing pre-computed audio features of each song. 
	- These features were pre-extracted (NO TEMPORAL MEANING) from the 30 or 60 second snippets, and capture timbre, chroma, and mfcc aspects of the audio. \
	- Each feature takes a continuous value. Type: float / continuous.
 

=======
Labels
=======

* trackID: unique id for each song (Maps features to their labels)
* genre: the genre label
	1. Soul and Reggae
	2. Pop
	3. Punk
	4. Jazz and Blues
	5. Dance and Electronica
	6. Folk
	7. Classic Pop and Rock
	8. Metal


"""