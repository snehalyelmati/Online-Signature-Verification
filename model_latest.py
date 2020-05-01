import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FOR NORMALIZING FEATURE VECTOR
feature_vector_main = pd.read_csv('features.csv')
feature_vector = feature_vector_main[
    ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
     'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure', 'max_pressure',
     'sample_points', 'sample_points_to_width', 'mean_pressure', 'pressure_variance', 'avg_x_velocity',
     'avg_y_velocity', 'max_x_velocity', 'max_y_velocity', 'samples_positive_x_velocity', 'samples_positive_y_velocity',
     'variance_x_velocity', 'variance_y_velocity', 'std_x_velocity', 'std_y_velocity', 'median_x_velocity',
     'median_y_velocity', 'corr_x_y_velocity', 'mean_x_acceleration', 'mean_y_acceleration', 'corr_x_y_acceleration',
     'variance_x_acceleration', 'variance_y_acceleration', 'std_x_acceleration', 'std_y_acceleration', 'x_local_minima',
     'y_local_minima']]

# min-max scaler
# feature_vector = ((feature_vector - feature_vector.min()) / (feature_vector.max() - feature_vector.min()))

# robust scaler
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
feature_vector = scaler.fit_transform(feature_vector)
feature_vector = pd.DataFrame(feature_vector)
feature_vector.columns = ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up',
                          'pen_ratio', 'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration',
                          'range_pressure', 'max_pressure', 'sample_points', 'sample_points_to_width', 'mean_pressure',
                          'pressure_variance', 'avg_x_velocity', 'avg_y_velocity', 'max_x_velocity', 'max_y_velocity',
                          'samples_positive_x_velocity', 'samples_positive_y_velocity', 'variance_x_velocity',
                          'variance_y_velocity', 'std_x_velocity', 'std_y_velocity', 'median_x_velocity',
                          'median_y_velocity', 'corr_x_y_velocity', 'mean_x_acceleration', 'mean_y_acceleration',
                          'corr_x_y_acceleration', 'variance_x_acceleration', 'variance_y_acceleration',
                          'std_x_acceleration', 'std_y_acceleration', 'x_local_minima', 'y_local_minima']

feature_vector['ID'] = feature_vector_main['ID'] - 1
feature_vector['F'] = feature_vector_main['F']

feature_vector.to_csv('features_normalized.csv', index=False)

feature_vector = pd.read_csv('features_normalized.csv')

cols = ['ID', 'avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up',
        'pen_ratio', 'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration',
        'range_pressure', 'max_pressure', 'sample_points', 'sample_points_to_width', 'mean_pressure',
        'pressure_variance', 'avg_x_velocity', 'avg_y_velocity', 'max_x_velocity', 'max_y_velocity',
        'samples_positive_x_velocity', 'samples_positive_y_velocity', 'variance_x_velocity',
        'variance_y_velocity', 'std_x_velocity', 'std_y_velocity', 'median_x_velocity',
        'median_y_velocity', 'corr_x_y_velocity', 'mean_x_acceleration', 'mean_y_acceleration',
        'corr_x_y_acceleration', 'variance_x_acceleration', 'variance_y_acceleration',
        'std_x_acceleration', 'std_y_acceleration', 'x_local_minima', 'y_local_minima', 'F']

num_of_signatures = 40  # number of classes
print('The number of signatures are:', num_of_signatures)
split_percentage = 0.8

# SHUFFLING DATASET
# TOTAL DATASET THAT IS TRAINED OR TESTED
total_df = feature_vector[:(40 * num_of_signatures)].sample(frac=1).reset_index(drop=True)

# TRAIN TEST SPLIT
train_data_count = int(0.8 * (40 * num_of_signatures))
train_df = total_df[:train_data_count]
test_df = total_df[train_data_count:(40 * num_of_signatures)]

x_train = train_df[
    ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
     'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]
y_train = train_df['ID']

x_test = test_df[
    ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
     'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]
y_test = test_df['ID']

# NEURAL NETWORK 1
model = keras.Sequential([
    keras.layers.Dense(4096, input_shape=[x_train.shape[1]]),
    keras.layers.Dense(2048, activation='relu', kernel_initializer='random_uniform'),
    keras.layers.Dense(2048, activation='relu', kernel_initializer='random_uniform'),
    keras.layers.Dense(1024, activation='relu', kernel_initializer='random_uniform'),
    keras.layers.Dense(1024, activation='relu', kernel_initializer='random_uniform'),
    keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform'),
    keras.layers.Dense(num_of_signatures, activation='softmax')
])

model.summary()

# optimizers
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
rms_prop = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

model.compile(optimizer=adam, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=150)

# testing using the test dataset
_, test_accuracy_1 = model.evaluate(x_test, y_test)
print('The accuracy of model 1:', test_accuracy_1)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

model.save('model.h5')

# neural network testing
model = keras.models.load_model('model.h5')


# # user 0
# user = np.argmax(model.predict([[0.41994994363790344, 0.5838235494969432, 0.30973598564406696, 0.4159338205667048,
#                                  0.19600740431574698, 0.02998012886030287, 0.04122178733324261, 7.48318065579558e-06,
#                                  0.013681592039800995, 0.9838449111470113, 0.11098039215686274, 0.42855771077395816,
#                                  0.46115218301018274, 0.2717245472746552, 0.08624898291293735, 0.5546218487394958]]))
# print('result:', user)


def model2_function(user):
    # TRAIN TEST SPLIT
    train2_count = 40 * split_percentage
    test2_count = 40 * (1 - split_percentage)

    train2_df = feature_vector[40 * user: int(40 * (user + 1) - test2_count)]
    test2_df = feature_vector[int(40 * (user + 1) - test2_count):40 * (user + 1)]

    x_train2 = train2_df[
        ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
         'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]
    y_train2 = train2_df['F']

    x_test2 = test2_df[
        ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
         'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]
    y_test2 = test2_df['F']

    # NEURAL NETWORK 2
    model = keras.Sequential([
        keras.layers.Dense(256, input_shape=[x_train.shape[1]]),
        keras.layers.Dense(256, activation='relu', kernel_initializer='random_uniform'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

    model.fit(x_train2, y_train2, epochs=100)

    # testing using the test dataset
    _, test_accuracy_2 = model.evaluate(x_test2, y_test2)

    print('The test accuracy of the model 2 is: ', test_accuracy_2)

    filename = './user_models/model2_' + str(user) + '.h5'
    model.save(filename)

# for i in range(num_of_signatures):
#     print('Model number', i, 'is under training...')
#     model2_function(i)
#     print()
#     print()

# # neural network 2 testing
# model2 = keras.models.load_model('model2.h5')
#
# forgery = model2.predict([[0.37860876400126514, 0.5842002702604662, 0.23375963359598828, 0.3731984471060051,
#                            0.15904421912184508, 0.01819863020149892, 0.034030628772419684, 3.7327241024737335e-06,
#                            0.004975124378109453, 0.9806138933764136, 0.11843137254901961, 0.37372767428461684,
#                            0.4172130004184684, 0.2649572731411967, 0.04905265605021504, 0.7154861944777912]])
# print('result 2:', int(round(forgery[0][0])))
