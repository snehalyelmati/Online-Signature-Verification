import math
from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
from tensorflow import keras

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/file_upload')
def file_upload():
    return render_template('file_upload.html')


def extract_features():
    V = []
    SDX = []
    SDY = []
    A = []
    SDV = []
    SDA = []


    file = pd.read_csv("uploadedFile.txt", delimiter=' ', names=['X', 'Y', 'TS', 'BS', 'AZ', 'AL', 'P'],
                     header=None,
                     skiprows=1)
    file_size = len(file)
    X = file['X']
    Y = file['Y']
    TS = file['TS']
    BS = file['BS']
    AZ = file['AZ']
    AL = file['AL']
    P = file['P']
    aX = sum(X) / file_size
    aY = sum(Y) / file_size
    for k in range(0, file_size - 1):
        if TS[k] == TS[k + 1]:
            X[k + 1] = (X[k] + X[k + 1]) / 2
            Y[k + 1] = (Y[k] + Y[k + 1]) / 2
            TS[k + 1] = (TS[k] + 1)
            BS[k + 1] = (BS[k] + BS[k + 1]) / 2
            AZ[k + 1] = (AZ[k] + AZ[k + 1]) / 2
            AL[k + 1] = (AL[k] + AL[k + 1]) / 2
            P[k + 1] = (P[k] + P[k + 1]) / 2
        if k < file_size:
            V.append(((math.sqrt((X[k + 1] - X[k]) ** 2 + (Y[k + 1] - Y[k]) ** 2)) * (TS[file_size - 1] - TS[0])) / (
                    TS[k + 1] - TS[k]))
        SDX.append((X[k] - aX) ** 2)
        SDY.append((Y[k] - aY) ** 2)
    SDX.append((X[file_size - 1] - aX) ** 2)
    SDY.append((Y[file_size - 1] - aY) ** 2)
    V.append(0)
    data = {'X': X, 'Y': Y, 'TS': TS, 'BS': BS, 'AZ': AZ, 'AL': AL, 'P': P, 'V': V, 'SDX': SDX, 'SDY': SDY}
    df = pd.DataFrame(data)

    file = df
    file_size = len(file)
    X = file['X']
    Y = file['Y']
    TS = file['TS']
    BS = file['BS']
    AZ = file['AZ']
    AL = file['AL']
    P = file['P']
    V = file['V']
    SDX = file['SDX']
    SDY = file['SDY']
    for k in range(0, file_size):
        if k < file_size - 1:
            A.append(((abs(V[k + 1] - V[k])) * (TS[file_size - 1] - TS[0])) / (TS[k + 1] - TS[k]))
    A.append(0)
    data = {'X': X, 'Y': Y, 'TS': TS, 'BS': BS, 'AZ': AZ, 'AL': AL, 'P': P, 'V': V, 'SDX': SDX, 'SDY': SDY, 'A': A}
    df = pd.DataFrame(data)

    file = df
    file_size = len(file)
    X = file['X']
    Y = file['Y']
    TS = file['TS']
    BS = file['BS']
    AZ = file['AZ']
    AL = file['AL']
    P = file['P']
    V = file['V']
    SDX = file['SDX']
    SDY = file['SDY']
    A = file['A']
    aV = sum(V) / file_size
    aA = sum(A) / file_size
    for k in range(0, file_size):
        SDV.append((V[k] - aV) ** 2)
        SDA.append((A[k] - aA) ** 2)
    data = {'X': X, 'Y': Y, 'TS': TS, 'BS': BS, 'AZ': AZ, 'AL': AL, 'P': P, 'V': V, 'SDX': SDX, 'SDY': SDY, 'A': A,
            'SDV': SDV, 'SDA': SDA}
    df = pd.DataFrame(data)

    avgX = []
    avgY = []
    avgSDX = []
    avgSDY = []
    avgV = []
    avgA = []
    avgSDV = []
    avgSDA = []
    pen_down = []
    pen_up = []
    pen_ratio = []
    sign_width = []
    sign_height = []
    width_height_ratio = []
    total_sign_duration = []
    range_pressure = []

    file = df
    file_size = len(file)
    X = file['X']
    Y = file['Y']
    TS = file['TS']
    BS = file['BS']
    AZ = file['AZ']
    AL = file['AL']
    P = file['P']
    V = file['V']
    SDX = file['SDX']
    SDY = file['SDY']
    A = file['A']
    SDV = file['SDV']
    SDA = file['SDA']

    avgX.append(sum(X) / file_size)
    avgY.append(sum(Y) / file_size)
    avgSDX.append(sum(SDX) / file_size)
    avgSDY.append(sum(SDY) / file_size)
    avgV.append(sum(V) / file_size)
    avgA.append(sum(A) / file_size)
    avgSDV.append(sum(SDV) / file_size)
    avgSDA.append(sum(SDA) / file_size)
    pen_down.append(sum(BS))
    pen_up.append(file_size - sum(BS))
    pen_ratio.append((sum(BS)) / (file_size - sum(BS)))
    sign_width.append(max(X) - min(X))
    sign_height.append(max(Y) - min(Y))
    width_height_ratio.append((max(X) - min(X)) / (max(Y) - min(Y)))
    total_sign_duration.append(TS[file_size - 1] - TS[0])
    range_pressure.append(max(P) - min(P))

    data = {'avgX': avgX, 'avgY': avgY, 'avgSDX': avgSDX, 'avgSDY': avgSDY, 'avgV': avgV, 'avgA': avgA,
            'avgSDV': avgSDV,
            'avgSDA': avgSDA, 'pen_down': pen_down, 'pen_up': pen_up, 'pen_ratio': pen_ratio, 'sign_width': sign_width,
            'sign_height': sign_height, 'width_height_ratio': width_height_ratio,
            'total_sign_duration': total_sign_duration, 'range_pressure': range_pressure}

    df = pd.DataFrame(data)

    dataset = pd.read_csv('Features.csv')
    dataset = dataset[
        ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
         'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]
    df = (df - dataset.min()) / (dataset.max() - dataset.min())

    return list(df.iloc[0])


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        if request.files:
            print('File uploaded is available.')
            signFile = request.files["signature_data"]
            signFile.save("uploadedFile.txt")

            try:
                features = extract_features()                
            except ZeroDivisionError as err:
                return render_template('404.html')

            print(features)

            # model 1 detects the id of the user of the signature
            model = keras.models.load_model('model.h5')

            user = np.argmax(model.predict([features]))

            print('user prediction result of the uploaded file:', user)

            # model 2 predicts if the signature is genuine or forged
            model2_name = './user_models/model2_' + str(user) + '.h5'
            model2 = keras.models.load_model(model2_name)

            forgery_status = model2.predict([features])
            print('forgery status:', int(round(forgery_status[0][0])))

            status = ""
            if int(round(forgery_status[0][0])) == 0:
                status = 'Genuine Signature for user ' + str(user+1)
                return render_template('genuine.html')
            else:
                status = 'Forged Signature for user ' + str(user+1)
                return render_template('forged.html')
            
    else:
        return render_template('prediction.html', prediction_text="File upload failed!")


if __name__ == "__main__":
    app.run(debug=True)
