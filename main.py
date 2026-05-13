import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ======================
# Dataset
# ======================
def generate_diabetes_dataset(n=500, noise_ratio=0.1, seed=42):
    np.random.seed(seed)

    half = n // 2
    noise_n = int(n * noise_ratio / 2)

    # ======================
    # Healthy (0)
    # ======================
    healthy = pd.DataFrame({
        "blood_sugar": np.random.randint(70, 125, half),
        "blood_pressure": np.random.randint(65, 90, half),
        "diabetes": 0
    })

    # ======================
    # Diabetic (1)
    # ======================
    diabetic = pd.DataFrame({
        "blood_sugar": np.random.randint(130, 220, half),
        "blood_pressure": np.random.randint(80, 125, half),
        "diabetes": 1
    })

    # ======================
    # Noise (mixing the classes)
    # ======================
    healthy_noise = pd.DataFrame({
        "blood_sugar": np.random.randint(120, 145, noise_n),
        "blood_pressure": np.random.randint(80, 100, noise_n),
        "diabetes": 0
    })

    diabetic_noise = pd.DataFrame({
        "blood_sugar": np.random.randint(105, 135, noise_n),
        "blood_pressure": np.random.randint(70, 95, noise_n),
        "diabetes": 1
    })

    # ======================
    # Combine
    # ======================
    df = pd.concat(
        [healthy, diabetic, healthy_noise, diabetic_noise],
        ignore_index=True
    )

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df

df = generate_diabetes_dataset()
# ==============================================================================================


# ======================
# Model
# ======================
X = df[["blood_sugar", "blood_pressure"]]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# ==============================================================================================

# ======================
# Sidebar
# ======================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    ["Dataset", "Model Performance", "Make Prediction"]
)

# ======================
# Page 1: Dataset
# ======================
if page == "Dataset":
    st.title("Dataset")

    st.subheader("Data")
    st.dataframe(df)

    st.subheader("Scatter Plot")

    st.scatter_chart(
        df,
        x="blood_sugar",
        y="blood_pressure",
        color="diabetes"
    )

# ======================
# Page 2: Performance
# ======================
elif page == "Model Performance":
    st.title("Model Performance")

    st.metric("Accuracy", f"{accuracy:.2f}")

# ======================
# Page 3: Prediction
# ======================
elif page == "Make Prediction":
    st.title("Predict Diabetes")

    blood_sugar = st.slider("Blood Sugar", 70, 220, 120)
    blood_pressure = st.slider("Blood Pressure", 60, 130, 80)

    if st.button("Predict"):
        input_data = np.array([[blood_sugar, blood_pressure]])
        prediction = knn.predict(input_data)[0]

        if prediction == 0:
            label = "Healthy" 
        else 
            label = "Diabetes"

        st.subheader(f"Prediction: {label}")

        # ======================
        # Add point to plot
        # ======================
        new_point = pd.DataFrame({
            "blood_sugar": [blood_sugar],
            "blood_pressure": [blood_pressure],
            "diabetes": [2]  # new category
        })

        plot_df = pd.concat([df, new_point], ignore_index=True)

        st.subheader("Visualization")

        st.scatter_chart(
            plot_df,
            x="blood_sugar",
            y="blood_pressure",
            color="diabetes"
        )
