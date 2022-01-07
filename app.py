from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


# Load Model
model = load_model('deployment_11102020')


def run():

    from PIL import Image

    st.sidebar.info('Python For Data Analysis Course– This app is created using PyCaret and Streamlit')

    st.title('Insurance Cost Prediction Model')

    # Capture
    age = st.number_input('Age', min_value = 1, max_value = 100, value = 21)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
    children = st.selectbox('Children', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    if st.checkbox('Smoker'):
        smoker = 'yes'
    else:
        smoker = 'no'

    region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

    output = ""

    # Input Dict
    input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
    input_df = pd.DataFrame([input_dict])

    # Predict
    if st.button('Predict'):
        output = predict_model(model, data = input_df)
        output = '$' + str(output['Label'][0])

    # Display
    st.success('The Insurance amount is {}'.format(output))
    
    image = Image.open('model_plot.png')

    st.image(image)
    
    image_3 = Image.open('feature-pycaret.png')

    st.image(image_3)
    
    image_2 = Image.open('residual-pycaret.png')

    st.image(image_2)
    

    
    # st.sidebar.info('Bar Chart and Line Graphs are just examples')

    # chart_data = pd.DataFrame(
    #  np.random.randn(50, 3),
    #  columns=["a", "b", "c"])

    # st.bar_chart(chart_data)
    
    # chart_data = pd.DataFrame(
    #  np.random.randn(20, 3),
    #  columns=['a', 'b', 'c'])

    # st.line_chart(chart_data)



if __name__ == '__main__':
    run()