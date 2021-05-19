import pandas as pd
from sklearn import tree
from sklearn import metrics
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.write("""
# Star Type Prediction App
This app predicts the **Star type**!
""")

video_file = open('Star.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

st.sidebar.header('User Input Parameters')

def user_input_features():
    magnitude = st.sidebar.slider('Absolute magnitude(Mv)', -11.92,  20.06, 5.4)
    radius = st.sidebar.slider('Radius(R/Ro)', 0.0084, 1948.5, 3.4)
    luminosity = st.sidebar.slider('Luminosity(L/Lo)', 0, 849420, 1000)
    spectral = st.sidebar.slider('Spectral Class', 0, 6, 4)
    temperature = st.sidebar.slider('Temperature (K)', 1939, 40000, 4000)
    data = {'Absolute magnitude(Mv)': magnitude,
            'Radius(R/Ro)':           radius,
            'Luminosity(L/Lo)':       luminosity,
            'Spectral Class':         spectral,
            'Temperature (K)':        temperature }
    features = pd.DataFrame(data, index=[0])
    return features

dataf = user_input_features()

st.markdown(' This app predicts the **Star type** based on the Hertzsprung-Russell Diagram, one of the most important tools in the study of stellar evolution. Developed independently in the early 1900s by Ejnar Hertzsprung and Henry Norris Russell, it plots the temperature of stars against their luminosity (the theoretical HR diagram), or the colour of stars (or spectral type) against their absolute magnitude.')



from PIL import Image
image = Image.open('HR_diagram.jpeg')
st.image(image, caption='HR diagram')




st.subheader('User Input parameters')
st.table(dataf)



############ Preparando os dados #############
df = pd.read_csv("estrelas.csv")

df = df.sample(frac=1).reset_index(drop=True)

df["Star color"] = pd.factorize(df["Star color"])[0]
df["Spectral Class"] = pd.factorize(df["Spectral Class"])[0]

cols = ["Absolute magnitude(Mv)","Radius(R/Ro)", "Luminosity(L/Lo)",
           "Spectral Class", "Temperature (K)"]

X = df[cols].values
y = df["Star type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

clf = tree.DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)

from PIL import Image
image = Image.open('tipos.png')
st.image(image, caption='Types')

st.subheader('This is the Prediction')
st.table(df["Star type"][y_pred].head(1))
