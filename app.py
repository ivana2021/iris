#!pip install streamlit scikit-learn pandas
# import library
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets

# load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# function for user input and prediction
def user_input():
    sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.0)
    sepal_width = st.slider('Sepal Width', 2.0, 4.5, 3.0)
    petal_length = st.slider('Petal Length', 1.0, 7.0, 4.0)
    petal_width = st.slider('Petal Width', 0.1, 2.5, 1.0)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# get user input
user_data = user_input()

# make prediction
prediction = model.predict(user_data)

# display the result
st.subheader('Prediction:')
st.write(iris.target_names[prediction[0]])

# display accuracy
st.subheader('Model Accuracy:')
st.write(f'The model accuracy on the test set is: {accuracy:.2%}')
