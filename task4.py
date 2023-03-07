import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# st.markdown(
#     """
#     <style>
#     .main{
#         background-color: #F5F5F5;
#     }
#     """,
#     unsafe_allow_html=True
# )

@st.cache
def get_data(filename):
    titanic_data=pd.read_csv(filename)
    return titanic_data


header=st.container()
dataset=st.container()
features=st.container()
modelTraining=st.container()

with header:
    st.title('Welcome to my awesome data science project')
    st.text('In this project I look into the survival probability of TITANIC disaster')


with dataset:
    st.header('Titanic dataset')
    st.text('I found this dataset on www.kaggle.com')
    titanic_data=get_data('Dataset/Titanic.csv')
    st.write(titanic_data.head(5))
    st.subheader('Suvived=1,Died=0 bar_plot')
    survived=pd.DataFrame(titanic_data['Survived'].value_counts())
    st.bar_chart(survived)


with features:
    st.header('The features I created')
    st.markdown('* **first feature:** I created this feature based on my EDA')

with modelTraining:
    le=LabelEncoder()
    # titanic_data=titanic_data.apply(LabelEncoder().fit_transform)
    titanic_data['Sex']=le.fit_transform(titanic_data['Sex'])
    titanic_data['Embarked']=le.fit_transform(titanic_data['Embarked'])

    st.header('Time to train the model!!!')
    st.text('Here you get to choose the hyperparameters of the model and see how the perfoemance changes')
    sel_col,disp_col=st.columns(2)
    max_depth=sel_col.slider('What should be the max depth of the model?',min_value=10,max_value=100,value=20,step=10)

    n_estimators=sel_col.selectbox('How many tress should be there?',options=[100,200,300,'No Limit'],index=0)
    sel_col.text('Here is a list of features in my data')
    sel_col.table(titanic_data.columns)

    input_feature=sel_col.text_input('Which feature should be used as input feature?','Fare')
    
    if n_estimators=='No Limit':
        reg=RandomForestClassifier(max_depth=max_depth)
    else:
        reg=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)

    X=titanic_data[[input_feature]]
    y=titanic_data[['Survived']]
    reg.fit(X,y)
    prediction=reg.predict(y)

    disp_col.subheader('Accuracy score od the model is:')
    disp_col.write(accuracy_score(y,prediction))



    


