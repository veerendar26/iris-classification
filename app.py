import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Classifier")

iris = load_iris()
X = iris.data
y = iris.target
clf = RandomForestClassifier()
clf.fit(X, y)

sepal_length = st.slider('Sepal Length', 4.0, 8.0)
sepal_width = st.slider('Sepal Width', 2.0, 4.5)
petal_length = st.slider('Petal Length', 1.0, 7.0)
petal_width = st.slider('Petal Width', 0.1, 2.5)

prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.write("### Predicted Species:", iris.target_names[prediction][0])
