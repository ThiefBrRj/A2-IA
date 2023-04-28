import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Carregando o modelo treinado
with open('modelo.pkl', 'rb') as arquivo:
    modelo = pickle.load(arquivo)

# Criando a interface da aplicação
st.title('Aplicação de predição')

# Recebendo os dados de entrada do usuário
valor_1 = st.number_input('Insira o valor 1:')
valor_2 = st.number_input('Insira o valor 2:')
valor_3 = st.number_input('Insira o valor 3:')

# Transformando os dados de entrada em um dataframe
dados = pd.DataFrame({'valor_1': [valor_1], 'valor_2': [valor_2], 'valor_3': [valor_3]})

# Realizando a predição com o modelo treinado
predicao = modelo.predict(dados)

# Exibindo o resultado da predição para o usuário
st.write(f'O valor previsto é: {predicao[0]}')
