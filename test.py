import math
from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
from tensorflow import keras
import pickle

app = Flask(__name__)
model = keras.models.load_model('model.h5')


@app.route('/')
def home():
    return render_template('index.html')


def extract_features(final_dataset):
    features = []

    distance = 0
    index = 0
    x_distance = 0
    y_distance = 0
    pressure = 0

    temp = 0
    time = 0

    tempX = []
    tempY = []
    tempP = []

    x = final_dataset.shape[0]

    for i in range(x):
        j = i - 1
        if j >= 0:
            distance += ((final_dataset['X'][j] - final_dataset['X'][i]) ** 2 + (
                        final_dataset['Y'][j] - final_dataset['Y'][i]) ** 2) ** 0.5

            x_distance += abs(final_dataset['X'][j] - final_dataset['X'][i])

            y_distance += abs(final_dataset['Y'][j] - final_dataset['Y'][i])

        pressure += final_dataset['P'][i]

    time = final_dataset['TS'][x - 1] - final_dataset['TS'][0]

    tempX = final_dataset['X'][:x]
    tempY = final_dataset['Y'][:x]
    tempP = final_dataset['P'][:x]

    # no_of_pts
    features.append(x)

    # pen-down
    features.append(final_dataset['T'][:x].sum())

    # pen_up
    features.append(x - final_dataset['T'][:x].sum())

    # pen_ratio
    features.append(float(final_dataset['T'][:x].sum() / (x - final_dataset['T'][:x].sum())))

    # velocity calculation
    features.append(float(distance / time))

    # x_velocity calculation
    features.append(float(x_distance / time))

    # y_velocity calculation
    features.append(float(y_distance / time))

    # avg_pressure calculation
    features.append(float(pressure / x))

    # avg_x calculation
    features.append(x_distance / x)

    # avg_y calculation
    features.append(y_distance / x)

    # std_dev_x calculation
    features.append(math.sqrt((x_distance - (x_distance / x)) ** 2 / x))

    # std_dev_y calculation
    features.append(math.sqrt((y_distance - (y_distance / x)) ** 2 / x))

    # acceleration calculation
    features.append(float(distance / (time * time)))

    # avg_vel calculation
    features.append(float(float(distance / time) / x))

    # avg_acc calculation
    features.append(float(float(distance / (time * time)) / x))

    # std_dev_vel calculation
    features.append((float(distance / time) - float(float(distance / time) / x)) / x)

    # std_dev_acc calculation
    features.append((float(distance / (time * time)) - float(float(distance / (time * time)) / x)) / x)

    # sign_width
    features.append(max(tempX) - min(tempX))

    # sign_height
    features.append(max(tempY) - min(tempY))

    # width_height-ratio
    features.append(float((max(tempX) - min(tempX)) / (max(tempY) - min(tempY))))

    # pts_width_ratio
    features.append(float(x / (max(tempX) - min(tempX))))

    # max_pressure
    features.append(max(tempP))

    # std_dev_pressure
    features.append(float((pressure - float(pressure / x)) ** 2) / x)

    # range_pressure
    features.append(max(tempP) - min(tempP))

    features = np.asarray(features, dtype=np.float32)
    features = ((features - features.min()) / (features.max() - features.min()))

    return features


@app.route('/predict', methods=["POST"])
def predict():
    name = ""
    if request.method == "POST":
        if request.files:
            print('File uploaded is available.')
            signFile = request.files["signature_data"]
            signFile.save("uploadedFile.txt")

            status = "This is the status of the prediction!"

            df = pd.read_csv("uploadedFile.txt", delimiter=' ', names=['X', 'Y', 'TS', 'T', 'AZ', 'AL', 'P'],
                             header=None,
                             skiprows=1)
            print('the dataframe from the file uploaded: ')
            print(df.head())

            features = extract_features(df)

            print(features[:10])
            print(features.shape)
            result = model.predict([[features]])

            print('The model summary:', model.summary())

            print('prediction result of the uploaded file:', result)
            print('result shape:', result.shape)
            print('max value of result:', np.amax(result))

            maxvalue_index = np.where(result == np.amax(result))
            print('index of the max value of result:', maxvalue_index[1][0])

            if maxvalue_index[1][0] % 2 == 0:
                status = 'Genuine Signature'
            else:
                status = 'Fake Signature'

            return render_template('prediction.html', prediction_text=status)
    else:
        return render_template('prediction.html', prediction_text="File upload failed!")


if __name__ == "__main__":
    app.run(debug=True)
