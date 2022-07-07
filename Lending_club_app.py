import streamlit as st
import numpy as np
import pandas as pd 
import pickle

st.write("""
# MSDE4 : Projet Classification Octroie de Crédit 
## by: LAAKAD Mouad 
Cette Application prédit si une personne pourra payer son crédit 

""")

st.sidebar.header('Veuillez insérer les parametres permettant d"évaluer Notre potentiel Client')
# Charger les modèles 
model=pickle.load(open("best_model.pkl",'rb'))
#######-------------------Lecture de données depuis Streamlit------------------#######
#--------------------------------------------------------------------------------------
def user_input_features():
    grades= ['B2', 'C4', 'C3', 'B1', 'C2', 'B3', 'A2', 'C1', 'A1', 'D5', 'A4','B4', 'B5', 'A5', 'E1', 'D2', 'D3', 'C5', 'D4', 'G1', 'E4', 'E5',
       'E3', 'E2', 'A3', 'D1', 'F1', 'G2', 'F2', 'F3', 'G3', 'G4', 'F4','G5', 'F5']
    sub_grade=st.sidebar.selectbox("Select  the Grade" ,grades)

    home_ownership_cat=['RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE']
    home_ownership=st.sidebar.selectbox(" Select the homeownership " ,home_ownership_cat)

    verification_status_cat=['Verified', 'Not Verified']
    verification_status=st.sidebar.selectbox(" verification_status : Verified or Not ? " ,verification_status_cat)

    purpose_cat=['credit_card', 'car', 'debt_consolidation', 'vacation', 'moving', 'medical', 'other', 'home_improvement', 'major_purchase','small_business', 'wedding', 'renewable_energy', 'house','educational']
    purpose=st.sidebar.selectbox(" Purpose? " ,purpose_cat)
     
    loan_amnt=st.sidebar.slider("loan_amnt",0,35000,11000,step=100)
    term=st.sidebar.selectbox("  term ? " ,[36,60])
    int_rate=st.sidebar.slider(" Interest Rate %",5,24,12 ,step=1)/100
    annual_inc=st.sidebar.slider(" Annual Income  ",1900,6000000,3000000,step=100)
    dti= st.sidebar.slider(" Debt-to-Income Ratio ",0,30,13,step=1)
    delinq_2yrs= st.sidebar.slider(" 30+ days delinquecy (last 2 years) ",0,13,5,step=1)
    fico_range_high=st.sidebar.slider(" Fico Score ",300,850,500,step=10)
    last_fico_range_high=st.sidebar.slider(" last Fico Score ",300,850,500,step=10) 
    inq_last_6mths=st.sidebar.slider("Number of inquiries in past 6 months",0,32,15,step=1)
    open_acc=st.sidebar.slider("Number of open credit lines in the borrower's credit file",0,50,25,step=1)
    pub_rec=st.sidebar.slider("Number of derogatory public records",0,10,5,step=1)
    revol_bal=st.sidebar.slider("Total credit revolving balance",0,1200000,600000,step=100) 
    revol_util=st.sidebar.slider("Revolving line utilization rate",0.00,1.20,0.60,step=0.01)
    total_acc=st.sidebar.slider("The total number of credit lines currently in the borrower's credit file",1,90,45,step=1)
    total_pymnt=st.sidebar.slider("Payments received to date for total amount funded",0,60000,30000,step=100)
    total_rec_prncp=st.sidebar.slider("Principal received to date",0,35000,12000,step=50) 
    total_rec_int= st.sidebar.slider("Interest received to date",0,24000,12000,step=50)
    total_rec_late_fee= st.sidebar.slider("Late fees received to date",0,210,100,step=2)
    recoveries= st.sidebar.slider("post charge off gross recovery",0,29000,14000,step=10)
    collection_recovery_fee=st.sidebar.slider("post charge off collection fee",0,7000,3500,step=20)
    last_pymnt_amnt=st.sidebar.slider("Last total payment amount received",0,36000,18000,step=100)

    data={'sub_grade': sub_grade,
            'home_ownership': home_ownership,
            'verification_status': verification_status,
             'purpose': purpose,
             'loan_amnt': loan_amnt,
             'term': term,
             'int_rate': int_rate,
             'annual_inc': annual_inc,
             'dti': dti,
             'delinq_2yrs': delinq_2yrs,
             'fico_range_high': fico_range_high,
             'inq_last_6mths': inq_last_6mths,
             'open_acc': open_acc,
             'pub_rec': pub_rec,
             'revol_bal': revol_bal,
             'revol_util': revol_util,
             'total_acc': total_acc,
             'total_pymnt': total_pymnt,
             'total_rec_prncp': total_rec_prncp,
             'total_rec_int': total_rec_int,
             'total_rec_late_fee': total_rec_late_fee,
             'recoveries': recoveries,
             'collection_recovery_fee': collection_recovery_fee,
             'last_pymnt_amnt': last_pymnt_amnt,
             'last_fico_range_high': last_fico_range_high }
    #features = pd.read_csv("test.csv")
    features =pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
#######-------------------Transformation de données pour Model------------------#######
#--------------------------------------------------------------------------------------

le_subgrade=pickle.load(open("encoder_sub_grade.pkl",'rb'))
le_home_ownership=pickle.load(open("encoder_home_ownership.pkl",'rb'))
le_verification_stat=pickle.load(open("encoder_verification_status.pkl",'rb'))
le_purpose=pickle.load(open("encoder_purpose.pkl",'rb'))
scaler=pickle.load(open("scaler.pkl",'rb'))
# fonction qui convertit une format yyyy-mm en timestamp

def transform_for_model(df):
    cat_data=df[["sub_grade","home_ownership" ,"verification_status","purpose"]]
    num_data=df[['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti',
       'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', 'open_acc',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
       'total_pymnt', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
       'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt',
       'last_fico_range_high']]
    cat_data["sub_grade"]=le_subgrade.transform(cat_data["sub_grade"])
    cat_data["home_ownership"]=le_home_ownership.transform(cat_data["home_ownership"])
    cat_data["verification_status"]=le_verification_stat.transform(cat_data["verification_status"])   
    cat_data["purpose"]= le_purpose.transform(cat_data["purpose"])
    
    df=pd.concat([cat_data,num_data],axis=1)
    df=scaler.transform(df)
    return df
#######-------------------Prédiction et affichage du résultat ------------------#######
#--------------------------------------------------------------------------------------
model=pickle.load(open("best_model.pkl",'rb'))

st.subheader('User Input parameters')
st.write(df)
df=transform_for_model(df)
if st.sidebar.button('Predict Solvability'):
  prediction = model.predict(df)
  prediction_proba = model.predict_proba(df)
  st.subheader('Class labels and their corresponding index number')
  st.write(pd.DataFrame(model.classes_))

  st.subheader('Prediction')
  dict_lend={0:"Probably would Not Pay" ,1: "Probably will pay"}
  prediction
  prediction_str=dict_lend[prediction[0]]
  st.write(prediction_str)
  st.subheader('Prediction Probability')
  st.write(prediction_proba)

