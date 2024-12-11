import pandas as pd 
import numpy as np
import torch
from torch import nn

# Interpolation of the absorption coefficient spectrum
def interp_exp(df, energy):
    return np.interp(energy, df.columns[1:].astype(float), df.iloc[0][1:].to_numpy().astype(float))

# NeuralNet architecture
D_in, H, D_out = 66, 100, 6
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.dropout = nn.Dropout(p=0.25)
        self.linear2 = nn.Linear(H, H)
        self.activation = nn.ReLU()
        self.linear3 = nn.Linear(H, D_out)
        self.softmax = nn.Softmax()
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

# Load data frame
df = pd.read_csv('Cu_MOR_df.csv')

# Unique channels
channel = np.unique(df['channel'])

# Load experimental data
df_exp = pd.read_csv('df_exp_test.csv')

# Interpolation of experimental data
df_exp_interp = interp_exp(df_exp, df.columns[7:].astype(float))

model = torch.load('CuMOR_model.pth') # Load the model
model.eval() # Set the model to evaluation mode

# Channel Prediction
with torch.no_grad():
    output = model(torch.FloatTensor((df_exp_interp)))
    
channel[np.argmax(output.numpy())]    
