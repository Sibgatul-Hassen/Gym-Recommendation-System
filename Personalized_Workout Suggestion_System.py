import streamlit as st
import pandas as pd
import joblib
import pickle


model = joblib.load("gym_recommender_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")
le_level = joblib.load("level_encoder.pkl")
le_goal = joblib.load("goal_encoder.pkl")
le_type = joblib.load("type_encoder.pkl")


with open("exercise_mapping.pkl", "rb") as f:
    exercise_mapping = pickle.load(f)
with open("unique_exercise_list_mapping.pkl", "rb") as f:
    unique_exercise_list_mapping = pickle.load(f)
with open("equipment_mapping.pkl", "rb") as f:
    equipment_mapping = pickle.load(f)
with open("equipment_list_mapping.pkl", "rb") as f:
    equipment_list_mapping = pickle.load(f)


reverse_exercise_id_to_list = {v: k for k,
                               v in unique_exercise_list_mapping.items()}
reverse_exercise_mapping = {v: k for k, v in exercise_mapping.items()}
reverse_equipment_id_to_list = {
    v: k for k, v in equipment_list_mapping.items()}
reverse_equipment_mapping = {v: k for k, v in equipment_mapping.items()}


def calculate_bmi(weight_kg, height_m):
    bmi = weight_kg / (height_m ** 2)
    return bmi


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal"
    elif 24.9 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"


def predict_gym_plan(sex, age, height, weight, hypertension, diabetes, goal, ftype):

    goal_encoded = le_goal.transform([goal])[0]
    type_encoded = le_type.transform([ftype])[0]

    input_df = pd.DataFrame([[
        sex, age, height, weight, hypertension, diabetes, goal_encoded, type_encoded
    ]], columns=['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'Fitness Goal', 'Fitness Type'])

    input_df_scaled = input_df.copy()
    input_df_scaled[["Age", "Height", "Weight"]] = scaler.transform(
        input_df[["Age", "Height", "Weight"]])

    prediction = model.predict(input_df_scaled)
    exercise_id, equipment_id = prediction[0]

    exercise_codes = reverse_exercise_id_to_list.get(exercise_id, [])
    exercise_names = [reverse_exercise_mapping.get(
        code, "Unknown") for code in exercise_codes]

    equipment_codes = reverse_equipment_id_to_list.get(equipment_id, [])
    equipment_names = [reverse_equipment_mapping.get(
        code, "Unknown") for code in equipment_codes]

    return exercise_names, equipment_names, round(calculate_bmi(weight, height), 2)


st.title("🏋️ Gym Recommendation System")
st.markdown(
    "Get personalized exercise & equipment recommendations based on your health profile.")


sex = st.radio("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

age = st.slider("Age", 10, 100, 25)
height = st.number_input("Height (in m)", min_value=0.5,
                         max_value=2.50, value=1.70)
weight = st.number_input(
    "Weight (in kg)", min_value=30.00, max_value=200.00, value=70.00)

hypertension = st.radio("Hypertension", ["No", "Yes"])
hypertension = 1 if hypertension == "Yes" else 0

diabetes = st.radio("Diabetes", ["No", "Yes"])
diabetes = 1 if diabetes == "Yes" else 0

goal = st.selectbox("Fitness Goal", le_goal.classes_)
ftype = st.selectbox("Fitness Type", le_type.classes_)


if st.button("🔍 Recommend Exercises & Equipment"):
    exercise_list, equipment_list, bmi = predict_gym_plan(
        sex, age, height, weight, hypertension, diabetes, goal, ftype
    )

    category = bmi_category(bmi)

    st.success(f"✅ Your BMI: **{bmi}** → Category: **{category}**")

    if category == "Underweight":
        st.warning(
            "⚠️ You are underweight. Consider focusing on muscle gain and a nutrient-rich diet.")
    elif category == "Normal":
        st.success(
            "🎯 You're in a healthy range! Maintain your current fitness plan.")
    elif category == "Overweight":
        st.info("📉 You're slightly overweight. Cardio and portion control may help.")
    elif category == "Obese":
        st.error(
            "🚨 Obesity detected. It's important to focus on regular workouts and diet control.")

    st.subheader("💪 Recommended Exercises:")
    st.write(", ".join(exercise_list))

    st.subheader("🛠️ Recommended Equipment:")
    st.write(", ".join(equipment_list))
# streamlit run WMS.py
