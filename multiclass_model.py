import math
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics

feature_vector = pd.read_csv('features_16.csv')
# print(feature_vector.head())

num_of_signatures = 40  # number of classes
print('The number of signatures are:', num_of_signatures)
split_percentage = 0.8

# SHUFFLING DATASET
# TOTAL DATASET THAT IS TRAINED OR TESTED
# total_df = feature_vector[:(40 * num_of_signatures)].sample(frac=1).reset_index(drop=True)

# TRAIN TEST SPLIT
# METHOD 1
# train_data_count = int(0.8 * (40 * num_of_signatures))
# train_df = total_df[:train_data_count]
# test_df = total_df[train_data_count:(40 * num_of_signatures)]

# METHOD 2
cols = ['no_of_pts', 'velocity', 'x_velocity', 'y_velocity', 'avg_pressure', 'avg_x', 'avg_y', 'pen_up',
        'pen_down', 'pen_ratio', 'std_dev_x', 'std_dev_y', 'acceleration', 'avg_velocity', 'avg_acceleration',
        'std_dev_vel', 'std_dev_acc', 'sign_width', 'sign_height', 'width_height_ratio', 'pts_width_ratio',
        'max_pressure', 'var_pressure', 'range_pressure', 'forgery']

train_df = pd.DataFrame(columns=cols)
test_df = pd.DataFrame(columns=cols)
feature_vector_temp = pd.DataFrame(columns=cols)

i = 0
while i < (num_of_signatures * 40):
    feature_vector_temp = feature_vector.iloc[i:i + 40].sample(frac=1).reset_index(drop=True)

    train_df = train_df.append(feature_vector.iloc[i:i + int(split_percentage * 40)])
    test_df = test_df.append(feature_vector.iloc[i + int(split_percentage * 40):(i + 40)])

    i += 40

print('shapes of train and test dataframes are: ', train_df.shape, test_df.shape)

# train_df.to_csv('train.csv', index=False)
# test_df.to_csv('test.csv', index=False)

x_train = train_df[
    ['no_of_pts', 'velocity', 'x_velocity', 'y_velocity', 'avg_pressure', 'avg_x', 'avg_y', 'pen_up', 'pen_down',
     'pen_ratio', 'std_dev_x', 'std_dev_y', 'acceleration', 'avg_velocity', 'avg_acceleration', 'std_dev_vel',
     'std_dev_acc', 'sign_width', 'sign_height', 'width_height_ratio', 'pts_width_ratio', 'max_pressure',
     'var_pressure', 'range_pressure']]
y_train = train_df['forgery']
y_train = y_train.astype('int')

x_test = test_df[
    ['no_of_pts', 'velocity', 'x_velocity', 'y_velocity', 'avg_pressure', 'avg_x', 'avg_y', 'pen_up', 'pen_down',
     'pen_ratio', 'std_dev_x', 'std_dev_y', 'acceleration', 'avg_velocity', 'avg_acceleration', 'std_dev_vel',
     'std_dev_acc', 'sign_width', 'sign_height', 'width_height_ratio', 'pts_width_ratio', 'max_pressure',
     'var_pressure', 'range_pressure']]
y_test = test_df['forgery']
y_test = y_test.astype('int')

# NEURAL NETWORK
model = keras.Sequential([
    keras.layers.Dense(4096, kernel_initializer='random_uniform', input_shape=[x_train.shape[1]]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2048, activation='relu', kernel_initializer='random_uniform'),
    keras.layers.Dense(2048, activation='relu', kernel_initializer='random_uniform'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024, activation='relu', kernel_initializer='random_uniform'),
    keras.layers.Dense(512, activation='relu', kernel_initializer='random_uniform'),
    keras.layers.Dense(num_of_signatures, activation='softmax')
])

model.summary()

model.compile(optimizer='SGD', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=250)

# testing using the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print('The test accuracy of the model is: ', test_accuracy)

model.save('model.h5')

# # RANDOM FORESTS
# from sklearn.ensemble import RandomForestClassifier
#
# clf = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=30, criterion="entropy", verbose=1)
# clf.fit(x_train, y_train)
#
# clf_predictions = clf.predict(x_test)
#
# print('Accuracy of random forest classifier is:', metrics.accuracy_score(y_test, clf_predictions))
#
# pickle.dump(clf, open('randomf_model.pkl', 'wb'))

# PREDICTION
# features below are of U1, U2, U3 respectively
# features = [
#     [6.3191153238546605, 822.7825942138707, 735.8255891530965, 894.552351831527, 610.555224256488, 787.6120097048474,
#      946.4889007293854, 750.0, 86.47450110864744, 213.70604147880974, 550.3575828571423, 440.9046197838143,
#      886.8251849664941, 921.8168104232308, 916.0619216501946, 922.0733544643379, 916.3288707867725, 373.72767428461685,
#      417.2130004184684, 264.9572731411967, 11.984528589860087, 776.4423076923077, 79.60285620306733, 715.4861944777912],
#     [211.69036334913113, 275.6170646724217, 377.5863308528875, 214.89498044941365, 708.6194216096023,
#      294.2361769622314, 152.55270110237453, 756.1576354679803, 225.05543237250555, 233.3797852282974,
#      355.6942054229762, 120.6757033829955, 171.59045861164466, 125.83433822684188, 72.63278995483971,
#      126.75510791947377, 73.17653665933622, 566.8331092759746, 341.60970846701076, 486.00343622595716,
#      99.73799346447949, 901.4423076923077, 278.24259617224357, 689.0756302521008],
#     [71.09004739336493, 523.2943231152137, 754.6492471730825, 353.79426045657675, 412.184061829692,
#      867.2146243038354, 409.1494200651386, 750.0, 131.9290465631929, 238.35287045386232, 775.8045712761351,
#      228.77925485069838, 368.10304612865326, 405.24452909150756, 260.9189891357819, 406.9135491400871,
#      262.0128167332722, 463.1265603994623, 436.18356814060536, 312.2764174976313, 39.5413625568408,
#      532.4519230769231, 63.19295410884063, 268.9075630252101],
#     [23.696682464454977, 795.3293805744283, 747.6676838392341, 856.8795076338873, 656.0027109404052, 806.7579501630601,
#      914.4295612253253, 750.0, 98.66962305986696, 220.3186053501653, 609.5923255372307, 458.1737426480164,
#      752.4122985385818, 787.0531211827997, 689.0499112220224, 788.3692109744311, 690.2076561599984, 411.7534088726714,
#      441.76314688241035, 273.8005567180695, 17.4333203820666, 903.8461538461538, 103.37778809688922, 902.7611044417768]
# ]
# users = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15', 'U16', 'U17',
#          'U18', 'U19', 'U20', 'U21', 'U22', 'U23', 'U24', 'U25', 'U26', 'U27', 'U28', 'U29', 'U30', 'U31', 'U32', 'U33',
#          'U34', 'U35', 'U36', 'U37', 'U38', 'U39', 'U40']
#
# # neural network testing
# model = keras.models.load_model('model_7layers_2dropouts.h5')
#
# result = model.predict(features)
# maxvalue_index = np.where(result == np.amax(result))
# print('index of the max value of result:', maxvalue_index)
#
# # random forest testing
# clf = pickle.load(open('randomf_model.pkl', 'rb'))
#
# result = clf.predict(features)
# print('The result of random forest prediction:', result)

######################################################################################################################
######################################################################################################################

# SVM CLASSIFIER
# supportvm = svm.SVC()
#
# supportvm.fit(x_train, y_train)
#
# result_svm = supportvm.predict(x_test.to_numpy())
#
# print('Accuracy of SVM using rbf:', metrics.accuracy_score(y_test, result_svm))
#
# # MLP MULTILAYER PERCEPTRON
# from sklearn.neural_network import MLPClassifier
#
# mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
#
# mlp.fit(x_train, y_train)
#
# result_mlp = mlp.predict(x_test.to_numpy())
#
# print(metrics.confusion_matrix(y_test, result_mlp))
#
# print('Accuracy of MLP using rbf:', metrics.accuracy_score(y_test, result_mlp))
#
# # DECISION TREE CLASSIFIER
# from sklearn.tree import DecisionTreeClassifier
#
# dtree_model = DecisionTreeClassifier(max_depth=10).fit(x_train, y_train)
# dtree_predictions = dtree_model.predict(x_test)
#
# print('Accuracy of decision tree classifier is:', metrics.accuracy_score(y_test, dtree_predictions))
#
