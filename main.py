import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ======================
# Generate dataset
# ======================
def generate_diabetes_dataset(n=500, noise_ratio=0.1, seed=42):
    np.random.seed(seed)
    half = n // 2
    noise_n = int(n * noise_ratio / 2)

    healthy = pd.DataFrame({
        "blood_sugar": np.random.randint(70, 125, half),
        "blood_pressure": np.random.randint(65, 90, half),
        "diabetes": 0
    })

    diabetic = pd.DataFrame({
        "blood_sugar": np.random.randint(130, 220, half),
        "blood_pressure": np.random.randint(80, 125, half),
        "diabetes": 1
    })

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

    df = pd.concat(
        [healthy, diabetic, healthy_noise, diabetic_noise],
        ignore_index=True
    )

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

df = generate_diabetes_dataset()

# ======================
# Train model
# ======================
X = df[["blood_sugar", "blood_pressure"]]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ======================
# UI - Sidebar
# ======================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Dataset", "Model Performance", "Make Prediction"]
)

# ======================
# Colors
# ======================
labels = {0: "Healthy", 1: "Diabetes"}
colors = {0: "#2ECC71", 1: "#E74C3C"}

# ======================
# Page 1: Dataset
# ======================
if page == "Dataset":
    st.title("Dataset Visualization")

    st.subheader("DataFrame")
    st.dataframe(df)

    st.subheader("Scatter Plot")

    fig, ax = plt.subplots()

    for label in [0, 1]:
        subset = df[df["diabetes"] == label]
        ax.scatter(
            subset["blood_sugar"],
            subset["blood_pressure"],
            label=labels[label],
            color=colors[label],
            alpha=0.7
        )

    ax.set_xlabel("Blood Sugar")
    ax.set_ylabel("Blood Pressure")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

# ======================
# Page 2: Model Performance
# ======================
elif page == "Model Performance":
    st.title("Model Performance")

    st.write(f"Accuracy: **{accuracy:.2f}**")

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

        st.subheader(f"Prediction: {labels[prediction]}")

        # Plot with new point
        fig, ax = plt.subplots()

        for label in [0, 1]:
            subset = df[df["diabetes"] == label]
            ax.scatter(
                subset["blood_sugar"],
                subset["blood_pressure"],
                color=colors[label],
                alpha=0.5
            )

        # New point
        ax.scatter(
            blood_sugar,
            blood_pressure,
            color="blue",
            s=150,
            label="New Patient",
            edgecolors="black"
        )

        ax.set_xlabel("Blood Sugar")
        ax.set_ylabel("Blood Pressure")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)
