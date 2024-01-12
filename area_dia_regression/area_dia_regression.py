import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_real_data_df():
    return pd.read_csv('real_data.csv', delimiter=';')

def load_real_data(scaler):
    data = load_real_data_df()
    scaled_data = scaler.transform(data)

    return torch.from_numpy(scaled_data[:, [0,2]]).float()


# Read train data as DataFrame
def load_train_data_df():
    return pd.read_csv('train_data.csv', delimiter=';')


# Load and transform train data into inputs and target tensors
def load_train_data(scaler):
    data = load_train_data_df()

    scaled_data = scaler.transform(data)
    inputs = torch.from_numpy(scaled_data[:, [0, 2]]).float()
    targets = torch.from_numpy(scaled_data[:, 1]).float().unsqueeze(1)

    return inputs, targets


def train_model(inputs, targets, learning_rate=0.01, max_epochs=500000):
    model = nn.Linear(2, 1).to('cpu')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup early stopping
    patience = 200
    min_val_loss = np.Inf
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        current_loss = loss.item()

        loss.backward()
        optimizer.step()


        print(f'E:{str(epoch + 1).zfill(len(str(max_epochs)))} L:{current_loss:0.8f}')

        if current_loss < min_val_loss:
            min_val_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break


    return model


def predict(model, scaler, data):
    model.eval()

    with torch.no_grad():
        predictions = model(data)
        dummy_array = np.zeros((predictions.shape[0], scaler.scale_.shape[0]))
        dummy_array[:, 1] = predictions.squeeze()

        real_scale_predictions = scaler.inverse_transform(dummy_array)

        return real_scale_predictions[:, 1]
    return None


def setup_scaler():
    ret = StandardScaler()
    df = load_train_data_df()
    ret.fit(df)

    return ret


if __name__ == '__main__':
    scaler = setup_scaler()
    inputs, targets = load_train_data(scaler)
    print("Training model...")
    model = train_model(inputs, targets, learning_rate=0.00001)

    print("Making predictions...")
    real_data = load_real_data(scaler)
    predictions = predict(model, scaler, real_data)

    result = load_real_data_df()
    result['conductor_dia'] = predictions
    print(result)

