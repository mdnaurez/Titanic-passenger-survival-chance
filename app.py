import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("passenger survival chance in the titanic journey")

pclass = st.slider('Enter the passenger class for the user',1,3)
sex = st.selectbox('Enter the passenger gender', ['male', 'female'])
sibsp = st.slider('Enter the passenger total number of sibling and spouse', 1,8)
parch = st.slider('Enter the passenger total number of parents and spouse', 1,8)
fare = st.number_input('Enter the fare of the passenger')
embarked = st.selectbox('Enter the passenger station from where they started the journey', ['southampton', 'chebourg', 'queenstown'])

data = pd.DataFrame([{'Pclass':pclass, 'Sex': sex, 'SibSp':sibsp, 'Parch':parch, 'Fare':fare, 'Embarked': embarked}])
# if st.button('Data'):
#     st.write(data)

model = load_model('model.h5')

with open('label_encoder.pkl', 'rb') as file:
    label = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

data['Sex'] = label.transform(data['Sex'])

embarked = one_hot_encoder.transform(data[['Embarked']])

embarked = pd.DataFrame(embarked, columns=one_hot_encoder.get_feature_names_out())

data = pd.concat([data.drop(columns=['Embarked']), embarked], axis = 1)

data[['Pclass', 'SibSp', 'Parch', 'Fare']] = scaler.transform(data[['Pclass', 'SibSp', 'Parch', 'Fare']])

y = model.predict(data)

y= y[0][0]

def chance(y):
    if(y>0.5):
        return 'the passenger will survive the journey'
    else:
        return 'the passenger will not survive the journey'

if st.button('predict survival chance'):
    st.write('probability of passenger survival chance', y)
    st.write(chance(y))