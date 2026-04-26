import pickle
import streamlit as st
import pandas as pd

# -------------------------------
# Load model + scaler
# -------------------------------
model = pickle.load(open('rf_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

# -------------------------------
# Get feature columns
# -------------------------------
model_columns = scaler.feature_names_in_

# -------------------------------
# UI
# -------------------------------
st.title("💼 Employee Attrition Prediction")

# -------- Numeric Inputs --------
Age = st.number_input('Age', 18, 60, 30)
DailyRate = st.number_input('Daily Rate', 100, 2000, 800)
DistanceFromHome = st.number_input('Distance From Home', 1, 50, 10)
Education = st.number_input('Education (1-5)', 1, 5, 3)
EnvironmentSatisfaction = st.number_input('Environment Satisfaction (1-4)', 1, 4, 3)
HourlyRate = st.number_input('Hourly Rate', 30, 150, 60)
JobInvolvement = st.number_input('Job Involvement (1-4)', 1, 4, 3)
JobLevel = st.number_input('Job Level (1-5)', 1, 5, 2)
JobSatisfaction = st.number_input('Job Satisfaction (1-4)', 1, 4, 3)
MonthlyIncome = st.number_input('Monthly Income', 1000, 20000, 5000)
MonthlyRate = st.number_input('Monthly Rate', 2000, 30000, 15000)
NumCompaniesWorked = st.number_input('Companies Worked', 0, 10, 2)
PercentSalaryHike = st.number_input('Salary Hike (%)', 10, 50, 15)
PerformanceRating = st.number_input('Performance Rating (1-4)', 1, 4, 3)
RelationshipSatisfaction = st.number_input('Relationship Satisfaction (1-4)', 1, 4, 3)
StockOptionLevel = st.number_input('Stock Option Level (0-3)', 0, 3, 1)
TotalWorkingYears = st.number_input('Total Working Years', 0, 40, 10)
TrainingTimesLastYear = st.number_input('Training Times Last Year', 0, 10, 3)
WorkLifeBalance = st.number_input('Work Life Balance (1-4)', 1, 4, 3)
YearsAtCompany = st.number_input('Years At Company', 0, 40, 5)
YearsInCurrentRole = st.number_input('Years In Current Role', 0, 20, 3)
YearsSinceLastPromotion = st.number_input('Years Since Last Promotion', 0, 15, 1)
YearsWithCurrManager = st.number_input('Years With Current Manager', 0, 20, 3)

# -------- Categorical --------
Gender = st.selectbox('Gender', ('Male','Female'))
OverTime = st.selectbox('OverTime', ('Yes','No'))
Department = st.selectbox('Department', ('Research & Development','Sales','Human Resources'))
MaritalStatus = st.selectbox('Marital Status', ('Single','Married','Divorced'))
EducationField = st.selectbox('Education Field', 
                             ('Life Sciences','Medical','Marketing','Technical Degree','Other'))
BusinessTravel = st.selectbox('Business Travel', 
                             ('Travel_Rarely','Travel_Frequently','Non-Travel'))
JobRole = st.selectbox('Job Role', 
                       ('Sales Executive','Research Scientist','Laboratory Technician',
                        'Manufacturing Director','Healthcare Representative',
                        'Manager','Sales Representative','Research Director','Human Resources'))

# -------------------------------
# Create full dataframe
# -------------------------------
input_data = pd.DataFrame(columns=model_columns)
input_data.loc[0] = 0  # initialize

# -------- Fill numeric --------
input_data['Age'] = Age
input_data['DailyRate'] = DailyRate
input_data['DistanceFromHome'] = DistanceFromHome
input_data['Education'] = Education
input_data['EnvironmentSatisfaction'] = EnvironmentSatisfaction
input_data['HourlyRate'] = HourlyRate
input_data['JobInvolvement'] = JobInvolvement
input_data['JobLevel'] = JobLevel
input_data['JobSatisfaction'] = JobSatisfaction
input_data['MonthlyIncome'] = MonthlyIncome
input_data['MonthlyRate'] = MonthlyRate
input_data['NumCompaniesWorked'] = NumCompaniesWorked
input_data['PercentSalaryHike'] = PercentSalaryHike
input_data['PerformanceRating'] = PerformanceRating
input_data['RelationshipSatisfaction'] = RelationshipSatisfaction
input_data['StockOptionLevel'] = StockOptionLevel
input_data['TotalWorkingYears'] = TotalWorkingYears
input_data['TrainingTimesLastYear'] = TrainingTimesLastYear
input_data['WorkLifeBalance'] = WorkLifeBalance
input_data['YearsAtCompany'] = YearsAtCompany
input_data['YearsInCurrentRole'] = YearsInCurrentRole
input_data['YearsSinceLastPromotion'] = YearsSinceLastPromotion
input_data['YearsWithCurrManager'] = YearsWithCurrManager

# -------- Binary --------
input_data['Gender'] = 1 if Gender == 'Male' else 0
input_data['OverTime'] = 1 if OverTime == 'Yes' else 0
input_data['Over18'] = 1  # always 1 in dataset

# -------- One-hot encoding --------

# Department
if Department == 'Research & Development':
    input_data['Department_Research & Development'] = 1
elif Department == 'Sales':
    input_data['Department_Sales'] = 1

# Marital Status
if MaritalStatus == 'Married':
    input_data['MaritalStatus_Married'] = 1
elif MaritalStatus == 'Single':
    input_data['MaritalStatus_Single'] = 1

# Education Field
if EducationField == 'Life Sciences':
    input_data['EducationField_Life Sciences'] = 1
elif EducationField == 'Marketing':
    input_data['EducationField_Marketing'] = 1
elif EducationField == 'Medical':
    input_data['EducationField_Medical'] = 1
elif EducationField == 'Other':
    input_data['EducationField_Other'] = 1
elif EducationField == 'Technical Degree':
    input_data['EducationField_Technical Degree'] = 1

# Business Travel
if BusinessTravel == 'Travel_Frequently':
    input_data['BusinessTravel_Travel_Frequently'] = 1
elif BusinessTravel == 'Travel_Rarely':
    input_data['BusinessTravel_Travel_Rarely'] = 1

# Job Role
input_data[f'JobRole_{JobRole}'] = 1

# -------------------------------
# Scaling
# -------------------------------
input_scaled = scaler.transform(input_data)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    # probability
    prob = model.predict_proba(input_scaled)[0][1]

    # display probability
    st.write(f"📊 Attrition Probability: {prob*100:.2f}%")

    # better decision logic
    if prob >= 0.5:
        st.error(f"⚠ High Risk of Employee Attrition ({prob*100:.2f}%)")
    elif prob >= 0.3:
        st.warning(f"⚠ Medium Risk of Employee Attrition ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Employee Attrition ({prob*100:.2f}%)")