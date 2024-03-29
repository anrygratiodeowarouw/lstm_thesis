import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import DataProcessor
import base64

st.title("Prediction of Load-settlement Curve of Pile Foundation using Deep Learning")
#membuat subheader
st.subheader("Thesis Project by: Anry Gratio Deo Warouw (25021069) Bandung Institute of Technology")
st.markdown("Selamat datang semua :partying_face:, saya Deo mahasiswa S2 Geoteknik sedang mengembangkan model AI :smile: dengan tujuan agar dapat digunakan untuk memprediksi Load-Settlement Tiang Bor di DKI Jakarta :satisfied:")

st.image('header.png')

st.subheader("Processing Soils Data")
st.text('The soil data used is the N-SPT data and the soil type is based on the borelog')

col1, col2,col3 = st.columns(3)

with col1:
    diameter = st.selectbox('Diameter (m)', options=[0.45, 0.5, 0.6, 0.8, 1.0, 1.2], index=0)
with col2:
    COL = st.number_input('COL (m)', value=0.00)
with col3:
    l = st.number_input('Total Length of Pile (m)', value=30.00)

# Load data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.text('Data yang diupload')
    st.write(df)

    data = DataProcessor(df, diameter, COL, l)

    data_used = data.get_data_used()

    st.text('Data yang digunakan')
    st.write(data_used)

    st.text('Hasil')
    st.write(data.get_all_value())

    out_df = pd.DataFrame(data.get_all_value(), index=[0])

    csv = out_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

st.subheader("Load-Settlement Curve Prediction using Deep Learning (Model 34)")
st.text('This model Trained by 60 data and tested by 11 data')
st.text('The model used is LSTM with 3 hidden layers and 20 hidden units')
st.text('RMSE Train: 519 kN, RMSE Test: 618 kN, R2 Train: 0.98, R2 Test: 0.96')
st.text('MAPE Train: 11.1%, MAPE Test: 13.1% ')
st.image('model34_11plot.png')
st.image('Performance.png')

st.text('Performa Model terhadap Jenis Tanah Ekivalen')
st.image('jenistanah.png')
st.text('Performa Model terhadap Lokasi di DKI Jakarta')
st.image('lokasi.png')
st.text('Sensitivitas Model dengan Partial Dependence Plots (PDP)/What-if Analysis dari Model LSTM')
st.image('sensitivity.png')



st.subheader("Input Data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selected_features = ['d', 'l', 's3', 's4', 'ns1', 'ns2', 's2', 'nt', 's1', 'ns3']

def get_ei(index):
    x = 0
    for i in range(index):
        x += 0.05*i
    return x

# Membuat cell RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Membuat cell LSTM

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.tanh(hidden)

        # Compute input gate
        input_gate = self.sigmoid(self.i2o(combined))

        # Compute output gate
        output_gate = self.sigmoid(self.i2o(combined))

        # Compute cell state
        cell = input_gate * self.tanh(cell) + (1 - input_gate) * self.tanh(self.i2h(combined))

        # Compute hidden state 
        hidden = output_gate * self.tanh(cell)

        # Compute output
        output = self.i2o(combined)

        return output, hidden, cell
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class Regressor(nn.Module):
    def __init__(self, feature_size, model_type, hidden_size, output_size, downscale=4, units = 1):
        super(Regressor, self).__init__()
        self.model_type = model_type
        self.units = units

        if downscale == 0:
            self.downscale_input = None
        else:
            self.downscale_input = nn.Linear(feature_size-1, downscale)

        if model_type == 'rnn':
            self.rnn = nn.ModuleList()
            if units > 1:
                for i in range(units):
                    if i == 0:
                        self.rnn.append(RNN(downscale+1 if downscale != 0 else feature_size, hidden_size, hidden_size))
                    elif i == units-1:
                        self.rnn.append(RNN(hidden_size, hidden_size, output_size))
                    else:
                        self.rnn.append(RNN(hidden_size, hidden_size, hidden_size))
            else:
                self.rnn.append(RNN(downscale+1 if downscale != 0 else feature_size, hidden_size, output_size))


        elif model_type == 'lstm':
            self.rnn = nn.ModuleList()
            if units > 1:
                for i in range(units):
                    if i == 0:
                        self.rnn.append(LSTM(downscale+1 if downscale != 0 else feature_size, hidden_size, hidden_size))
                    elif i == units-1:
                        self.rnn.append(LSTM(hidden_size, hidden_size, output_size))
                    else:
                        self.rnn.append(LSTM(hidden_size, hidden_size, hidden_size))
            else:
                self.rnn.append(LSTM(downscale+1 if downscale != 0 else feature_size, hidden_size, output_size))

        elif model_type == 'gru':
            self.rnn = nn.ModuleList()
            if units > 1:
                for i in range(units):
                    if i == 0:
                        self.rnn.append(GRU(downscale+1 if downscale != 0 else feature_size, hidden_size, hidden_size))
                    elif i == units-1:
                        self.rnn.append(GRU(hidden_size, hidden_size, output_size))
                    else:
                        self.rnn.append(GRU(hidden_size, hidden_size, hidden_size))
            else:
                self.rnn.append(GRU(downscale+1 if downscale != 0 else feature_size, hidden_size, output_size))

    def forward(self, input):
        batch_size = input.shape[0]
        if self.downscale_input is not None:
            feature1 = input[:,:,:-1]
            feature2 = input[:,:,-1].unsqueeze(2)
            feature1 = self.downscale_input(feature1)
            input = torch.cat((feature1, feature2), dim=2)
        hidden = []
        for i in range(len(self.rnn)):
            hidden.append(self.rnn[i].init_hidden(batch_size).to(device))
        if self.model_type == 'lstm':
            cell = []
            for i in range(len(self.rnn)):
                cell.append(self.rnn[i].init_hidden(batch_size).to(device))
        output = torch.zeros(input.shape[0], input.shape[1], 1)
        for i in range(input.shape[1]): #iterate trough sequence
            if self.model_type == 'rnn' or self.model_type == 'gru':
                if self.units > 1:
                    for j in range(len(self.rnn)):
                        if j == 0:
                            out, hidden[j] = self.rnn[j](input[:,i,:], hidden[j])
                        elif j == len(self.rnn)-1:
                            output[:,i,:], hidden[j] = self.rnn[j](out, hidden[j])
                        else:
                            out, hidden[j] = self.rnn[j](out, hidden[j])
                else:
                    output[:,i,:], hidden[0] = self.rnn[0](input[:,i,:], hidden[0])

            elif self.model_type == 'lstm':
                if self.units > 1:
                    for j in range(len(self.rnn)):
                        if j == 0:
                            out, hidden[j], cell[j] = self.rnn[j](input[:,i,:], hidden[j], cell[j])
                        elif j == len(self.rnn)-1:
                            output[:,i,:], hidden[j], cell[j] = self.rnn[j](out, hidden[j], cell[j])
                        else:
                            out, hidden[j], cell[j] = self.rnn[j](out, hidden[j], cell[j])
                else:
                    output[:,i,:], hidden[0], cell[0] = self.rnn[0](input[:,i,:], hidden[0], cell[0])
                    
        return output

scaler = pickle.load(open('15July/scaler_iter_13_scaler_hs_20_unit_3.pkl', 'rb'))    

regressor = Regressor(model_type='lstm', feature_size=len(selected_features)+1, 
                      hidden_size=20, output_size=1, downscale=0, units=3)

regressor.load_state_dict(torch.load('15July/model_iter_13_model_best_categorical_hs_20_unit_3.pth', map_location=device))

regressor.eval()

def preprocessing(data, scaler):
    ei = data[:,-1]
    x = data
    x = np.expand_dims(x, axis=0)
    x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    return x, ei

def inference(model, x, ei):
    # search index of ei that has value 0
    index = np.where(ei == 0)[0][0]

    x = torch.from_numpy(x).float().to(device)
    # memastikan model berada di perangkat yang sama dengan tensor x
    model = model.to(device)
    y_pred = model(x)
    y_pred = y_pred.squeeze(0).detach().cpu().numpy()
    y_pred = y_pred*10000
    #remove value below 0
    y_pred[y_pred < 0] = 0

    # Change value of y_pred that has index below index of ei that has value 0
    y_pred[index] = 0
    return y_pred


import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


def plot_by_ei(data):
    x, ei = preprocessing(data, scaler)
    print(x)
    y_pred = inference(regressor, x, ei)
    plt.figure(figsize=(15,15))
    plt.plot(y_pred, ei, label='Prediction', color='red', linewidth=3)
    #title = 'Prediction of Load-settlement Curve of Pile Foundation using Deep Learning'
    # Font Family
    plt.rcParams['font.family'] = 'Arial'
    # membuat jarak antara title dengan grafik
    plt.title('Prediction of Load-settlement Curve of Pile Foundation using Deep Learning', fontsize=24, fontweight='bold', pad=30.0)
    plt.xlabel('Load at Pile Head (kN)', fontsize=20)
    plt.ylabel('Pile Top Settlement (% Diameter)', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0)
    plt.ylim(0, max(ei+1))

    plt.gca().invert_yaxis()
    # mengatur garis grid pada grafik
    plt.grid(linestyle='--', linewidth=1, alpha=0.5)
    plt.legend(fontsize=14)

    return plt, y_pred

# Example usage:
# plot_by_ei(data)



#create 10 columns
#col1, col2, col3, col4 = st.columns(4)
#col5, col6, col7 = st.columns(3)
#col8, col9, col10, col11 = st.columns(4)

s_value = {
    'sand': 1,
    'silt': 2,
    'clay': 3,
}

#input data
#with col1:
x1 = st.selectbox('Diameter Tiang (m)', options=[0.8, 1.0, 1.2], index=0)

#with col2:
x2 = st.number_input('Panjang Efektif Tiang (m) (Min=16.5; Max=68.05)', min_value=16.5, max_value=68.05, value=30., step=1.)

#with col3:
x3 = st.number_input('N-SPT rata-rata di Selimut Tiang Bagian 1 (Ns1) (Min=3; Max=79)', min_value=3., max_value=79., value=16., step=1.)

#with col4:
x4 = st.selectbox('Jenis Tanah Ekivalen di Selimut Tiang Bagian 1 (S1)', options=list(s_value.keys()), index=0)
x4 = s_value[x4]

#with col5:
x5 = st.number_input('N-SPT rata-rata di Selimut Tiang Bagian 2 (Ns2) (Min=8.6; Max=108)', min_value=8.6, max_value=108.1, value=45., step=1.)

#with col6:
x6 = st.selectbox('Jenis Tanah Ekivalen di Selimut Tiang Bagian 2 (S2)', options=list(s_value.keys()), index=0)
x6 = s_value[x6]

#with col7:
x7 = st.number_input('N-SPT rata-rata di Selimut Tiang Bagian 3 (Ns3) (Min=21; Max=104)', min_value=21., max_value=104.2, value=43., step=1.)

#with col8:
x8 = st.selectbox('Jenis Tanah Ekivalen di Selimut Tiang Bagian 3 (S3)', options=list(s_value.keys()), index=0)
x8 = s_value[x8]

#with col9:
x9 = st.number_input('N-SPT rata-rata di Ujung Tiang (Ntip) (Min=16.7; Max=180)', min_value=16.7, max_value=180., value=53., step=1.)

#with col10:
x10 = st.selectbox('Jenis Tanah Ekivalen di Ujung Tiang (S4)', options=list(s_value.keys()), index=0)
x10 = s_value[x10]

#with col11:
x11 = st.number_input('Jumlah Data Penurunan (pers. get_ei)', min_value=0,max_value=20, value=13, step=1)

print(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)


if x1==0 or x2==0 or x3==0 or x4==0 or x5==0 or x6==0 or x7==0 or x8==0 or x9==0 or x10==0 or x11==0:
    st.write('Please input the data')
    st.stop()
# change x to array

#ini ngurutinnya manual
current_feature = ['d', 'l', 'ns1', 's1', 'ns2', 's2', 'ns3', 's3', 'nt', 's4']
selected_features = ['d', 'l', 's3', 's4', 'ns1', 'ns2', 's2', 'nt', 's1', 'ns3']

# Urutin secara manual
x = [x1, x2, x8, x10, x3, x5, x6, x9, x4, x7]

# change to array
x = np.array(x)

# add ei 
x = [np.append(x.copy(), get_ei(i+1)) for i in range(x11)]

x = np.array(x)
print(x)

# Preprocessing the X
plot, y_pred = plot_by_ei(x)

#st.text(f'Prediction (kN): {y_pred}')

st.pyplot(plot)

st.markdown("**Disclaimer: Model hanya dapat digunakan dalam range input yang sudah ditentukan**", unsafe_allow_html=True)
st.markdown("**Mohon digunakan dengan hati-hati!**", unsafe_allow_html=True)


st.markdown("Credit to:")
st.markdown("            P2MI ITB")
st.markdown("Thesis Project under supervision of:")
st.markdown(" Ir. Erza Rismantojo, Ph.D")
st.text("")
st.text("")
st.text("")

st.markdown(":smile: Feel free to contact me : 25021069@mahasiswa.itb.ac.id, Github: https://github.com/anrygratiodeowarouw")


