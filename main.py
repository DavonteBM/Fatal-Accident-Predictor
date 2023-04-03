import pandas
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Read data into pandas
df = pandas.read_csv('ksi.csv')


#Data cleaning
#Removing irrelevant features 
df.drop(['X', 'Y', 'HOOD_ID',  'NEIGHBOURHOOD', 'INDEX_', 'YEAR', 'DATE','TIME', 'HOUR', 'STREET1', 'STREET2', 'OFFSET', 'ROAD_CLASS', 'DISTRICT', 'WARDNUM', 'DIVISION', 'LATITUDE', 'LONGITUDE', 'LOCCOORD', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'FATAL_NO', 'INITDIR', 'DRIVACT', 'DRIVCOND','PEDTYPE', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND', 'POLICE_DIVISION', 'HOOD_ID', 'PEDESTRIAN', 'CYCLIST',  'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'PASSENGER', 'EMERG_VEH'], inplace=True, axis=1)

#Dealing with null values. Replace with zero
df['ACCNUM'] = df['ACCNUM'].replace(['<Null>'], '0')
df['ACCLASS'] = df['ACCLASS'].replace(['<Null>'], '0')
df['IMPACTYPE'] = df['IMPACTYPE'].replace(['<Null>'], '0')
df['INVTYPE'] = df['INVTYPE'].replace(['<Null>'], '0')
df['INVAGE'] = df['INVAGE'].replace(['<Null>'], '0')
df['INJURY'] = df['INJURY'].replace(['<Null>'], '0')
df['VEHTYPE'] = df['VEHTYPE'].replace(['<Null>'], '0')
df['MANOEUVER'] = df['MANOEUVER'].replace(['<Null>'], '0')
df['SPEEDING'] = df['SPEEDING'].replace(['<Null>'], '0')
df['AG_DRIV'] = df['AG_DRIV'].replace(['<Null>'], '0')
df['REDLIGHT'] = df['REDLIGHT'].replace(['<Null>'], '0')
df['ALCOHOL'] = df['ALCOHOL'].replace(['<Null>'], '0')
df['DISABILITY'] = df['DISABILITY'].replace(['<Null>'], '0')

#Encode features 
label_encoder = preprocessing.LabelEncoder()
df['ACCLASS']= label_encoder.fit_transform(df['ACCLASS'])
df['IMPACTYPE']= label_encoder.fit_transform(df['IMPACTYPE'])
df['INVTYPE']= label_encoder.fit_transform(df['INVTYPE'])
df['INVAGE']= label_encoder.fit_transform(df['INVAGE'])
df['INJURY']= label_encoder.fit_transform(df['INJURY'])
df['VEHTYPE']= label_encoder.fit_transform(df['VEHTYPE'])
df['MANOEUVER']= label_encoder.fit_transform(df['MANOEUVER'])
df['SPEEDING']= label_encoder.fit_transform(df['SPEEDING'])
df['AG_DRIV']= label_encoder.fit_transform(df['AG_DRIV'])
df['REDLIGHT']= label_encoder.fit_transform(df['REDLIGHT'])
df['ALCOHOL']= label_encoder.fit_transform(df['ALCOHOL'])
df['DISABILITY']= label_encoder.fit_transform(df['DISABILITY'])


#Create model
features = ['IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY', 'VEHTYPE', 'MANOEUVER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']
X = df[features]
y = df['ACCLASS']

#Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Fit Decision Tree Classifier to data
classifier = DecisionTreeClassifier(criterion = "entropy", random_state=0)
classifier.fit(X_train, y_train)

#Predict
y_pred = classifier.predict(X_test)

# Evaluate accuracy of the model
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
accuracy = (metrics.accuracy_score(y_test, y_pred) *(100))
print("Accuracy score:",accuracy)