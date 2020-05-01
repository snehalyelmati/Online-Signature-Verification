import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import os
from collections import Counter

pd.set_option('display.max_columns', None)

# FOR NORMALIZING FEATURE VECTOR
feature_vector_main = pd.read_csv('./Features.csv')
feature_vector = feature_vector_main[
    ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up',
     'pen_ratio', 'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration',
     'range_pressure', 'max_pressure', 'sample_points', 'sample_points_to_width', 'mean_pressure',
     'pressure_variance', 'avg_x_velocity', 'avg_y_velocity', 'max_x_velocity', 'max_y_velocity',
     'samples_positive_x_velocity', 'samples_positive_y_velocity', 'variance_x_velocity',
     'variance_y_velocity', 'std_x_velocity', 'std_y_velocity', 'median_x_velocity',
     'median_y_velocity', 'corr_x_y_velocity', 'mean_x_acceleration', 'mean_y_acceleration',
     'corr_x_y_acceleration', 'variance_x_acceleration', 'variance_y_acceleration',
     'std_x_acceleration', 'std_y_acceleration', 'x_local_minima', 'y_local_minima']]

# scaler = RobustScaler()
# feature_vector = scaler.fit_transform(feature_vector)
# feature_vector = pd.DataFrame(feature_vector)
# feature_vector.columns = ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up',
#                           'pen_ratio', 'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration',
#                           'range_pressure', 'max_pressure', 'sample_points', 'sample_points_to_width', 'mean_pressure',
#                           'pressure_variance', 'avg_x_velocity', 'avg_y_velocity', 'max_x_velocity', 'max_y_velocity',
#                           'samples_positive_x_velocity', 'samples_positive_y_velocity', 'variance_x_velocity',
#                           'variance_y_velocity', 'std_x_velocity', 'std_y_velocity', 'median_x_velocity',
#                           'median_y_velocity', 'corr_x_y_velocity', 'mean_x_acceleration', 'mean_y_acceleration',
#                           'corr_x_y_acceleration', 'variance_x_acceleration', 'variance_y_acceleration',
#                           'std_x_acceleration', 'std_y_acceleration', 'x_local_minima', 'y_local_minima']

feature_vector = ((feature_vector - feature_vector.min()) / (feature_vector.max() - feature_vector.min()))

feature_vector['ID'] = feature_vector_main['ID'] - 1
feature_vector['F'] = feature_vector_main['F']

feature_vector.to_csv('features_normalized.csv', index=False)

feature_vector = pd.read_csv('features_normalized.csv')

num_of_signatures = 40  # number of classes
print('The number of signatures are:', num_of_signatures)
split_percentage = 0.2

# os.getcwd()
# os.mkdir('user_models')
# os.chdir('user_models/')

cols = ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up',
        'pen_ratio', 'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration',
        'range_pressure', 'max_pressure', 'sample_points', 'sample_points_to_width', 'mean_pressure',
        'pressure_variance', 'avg_x_velocity', 'avg_y_velocity', 'max_x_velocity', 'max_y_velocity',
        'samples_positive_x_velocity', 'samples_positive_y_velocity', 'variance_x_velocity',
        'variance_y_velocity', 'std_x_velocity', 'std_y_velocity', 'median_x_velocity',
        'median_y_velocity', 'corr_x_y_velocity', 'mean_x_acceleration', 'mean_y_acceleration',
        'corr_x_y_acceleration', 'variance_x_acceleration', 'variance_y_acceleration',
        'std_x_acceleration', 'std_y_acceleration', 'x_local_minima', 'y_local_minima']

user_accuracy = {}

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                            patience=7,
                                                            verbose=1,
                                                            factor=0.25,
                                                            min_lr=0.000001)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0,
                                                  verbose=1, mode='auto', restore_best_weights=False)


def model2_function(user):
    model_2_df = feature_vector[user * 40:(user + 1) * 40].copy()
    model_2_df = model_2_df.sample(frac=1, random_state=1).reset_index(drop=True)

    y = model_2_df['F']
    X = model_2_df.drop(['ID', 'F'], axis=1)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=split_percentage, shuffle=True, random_state=1)

    # NEURAL NETWORK 2
    model = keras.Sequential([
        keras.layers.Dense(256, input_shape=[x_train.shape[1]]),
        #         keras.layers.Dense(1024, activation='relu', kernel_initializer='random_uniform'),
        #         keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform'),
        #         keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform'),
        keras.layers.Dense(256, activation='relu', kernel_initializer='random_uniform'),
        #         keras.layers.Dense(256, activation='relu', kernel_initializer='random_uniform'),
        keras.layers.Dense(128, activation='relu', kernel_initializer='random_uniform'),
        #         keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform'),
        #         keras.layers.Dense(64, activation='relu', kernel_initializer='random_uniform'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=100, verbose=0, callbacks=[learning_rate_reduction])

    # summarize history for train accuracy
    #     plt.plot(history.history['accuracy'])
    #     plt.plot(history.history['val_accuracy'])
    #     plt.title('model accuracy')
    #     plt.ylabel('accuracy')
    #     plt.xlabel('epoch')
    #     plt.legend(['train', 'test'], loc='upper left')
    #     plt.show()

    # testing using the test dataset
    _, test_accuracy_2 = model.evaluate(x_val, y_val)

    user_accuracy[user] = test_accuracy_2

    filename = './user_models/model2_' + str(user) + '.h5'
    model.save(filename)


#     print('The test accuracy of the model 2 user', user, 'is: ', test_accuracy_2)


# second neural nets creation loop
for i in range(num_of_signatures):
    print('Model number', i, 'is under training...')
    model2_function(i)

lists = sorted(user_accuracy.items())  # sorted by key, return a list of tuples
x, y = zip(*lists)  # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.show()

UAL = []
for k, v in user_accuracy.items():
    UAL.append(v)
    if v < .9:
        print(k, v)

print('mean:', np.mean(UAL))

print('counter:', Counter(UAL))

# # neural network 2 testing
# model2 = keras.models.load_model('model2.h5')

# forgery = model2.predict([[0.37860876400126514, 0.5842002702604662, 0.23375963359598828, 0.3731984471060051,
#                            0.15904421912184508, 0.01819863020149892, 0.034030628772419684, 3.7327241024737335e-06,
#                            0.004975124378109453, 0.9806138933764136, 0.11843137254901961, 0.37372767428461684,
#                            0.4172130004184684, 0.2649572731411967, 0.04905265605021504, 0.7154861944777912]])
# print('result 2:', int(round(forgery[0][0])))
