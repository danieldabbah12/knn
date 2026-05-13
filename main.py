import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ── Data ──────────────────────────────────────────────────────────────────────

np.random.seed(42)

healthy  = pd.DataFrame({"blood_sugar": np.random.randint(70,  125, 250),
                          "blood_pressure": np.random.randint(65,  90,  250),
                          "diabetes": 0})

diabetic = pd.DataFrame({"blood_sugar": np.random.randint(130, 220, 250),
                          "blood_pressure": np.random.randint(80,  125, 250),
                          "diabetes": 1})

df = pd.concat([healthy, diabetic]).sample(frac=1, random_state=42).reset_index(drop=True)

# ── Model ─────────────────────────────────────────────────────────────────────

X = df[["blood_sugar", "blood_pressure"]]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ── App ───────────────────────────────────────────────────────────────────────

st.title("🩺 Diabetes KNN Classifier")

page = st.sidebar.radio("Go to", ["Data", "Performance", "Predict"])

# ── Page 1 : Data ─────────────────────────────────────────────────────────────

if page == "Data":
    st.header("Dataset")
    st.dataframe(df)

    st.header("Scatter Plot")
    st.scatter_chart(df, x="blood_sugar", y="blood_pressure", color="diabetes")

# ── Page 2 : Performance ──────────────────────────────────────────────────────

elif page == "Performance":
    st.header("Model Performance")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")

# ── Page 3 : Predict ──────────────────────────────────────────────────────────

elif page == "Predict":
    st.header("Patient Diagnosis")

    blood_sugar    = st.number_input("Blood Sugar Level (mg/dL)", min_value=40, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mmHg)",     min_value=40, max_value=160, value=80)

    if st.button("Diagnose"):
        patient = pd.DataFrame({"blood_sugar": [blood_sugar], "blood_pressure": [blood_pressure]})
        prediction = knn.predict(patient)[0]

        if prediction == 1:
            st.error("🔴 Diabetic")
        else:
            st.success("🟢 Healthy")

        st.header("Nearest Neighbours on the Map")
        _, indices = knn.kneighbors(patient)
        neighbours = df.iloc[indices[0]].copy()

        plot_df = df.copy()
        patient["diabetes"] = prediction

        combined = pd.concat([plot_df, patient])
        st.scatter_chart(combined, x="blood_sugar", y="blood_pressure", color="diabetes")

        st.subheader("Neighbours")
        st.dataframe(neighbours[["blood_sugar", "blood_pressure", "diabetes"]])
