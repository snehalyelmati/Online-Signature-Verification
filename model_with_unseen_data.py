import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

# FOR NORMALIZING FEATURE VECTOR
feature_vector_main = pd.read_csv('features.csv')
feature_vector = feature_vector_main[
    ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
     'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]

feature_vector = ((feature_vector - feature_vector.min()) / (feature_vector.max() - feature_vector.min()))

feature_vector['ID'] = feature_vector_main['ID'] - 1
feature_vector['F'] = feature_vector_main['F']

feature_vector.to_csv('features_16.csv', index=False)

feature_vector = pd.read_csv('features_16.csv')

cols = ['ID', 'avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
        'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure', 'F']

num_of_signatures = 40  # number of classes
print('The number of signatures are:', num_of_signatures)
split_percentage = 0.8

# SHUFFLING DATASET
# TOTAL DATASET THAT IS TRAINED OR TESTED
total_df = feature_vector[:(40 * num_of_signatures)].sample(frac=1).reset_index(drop=True)
total_df.to_csv('total_df.csv', index=False)

# TRAIN TEST SPLIT
train_data_count = int(0.8 * (40 * num_of_signatures))
train_df = total_df[:train_data_count]
test_df = total_df[train_data_count:(40 * num_of_signatures) - 20]

print('train df shape:', train_df.shape)
print('test df shape:', test_df.shape)

x_train = train_df[
    ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
     'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]
y_train = train_df['ID']

x_test = test_df[
    ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
     'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]
y_test = test_df['ID']

# NEURAL NETWORK 1
# model = keras.Sequential([
#     keras.layers.Dense(2048, input_shape=[x_train.shape[1]]),
#     # keras.layers.Dropout(0.5),
#     keras.layers.Dense(2048, activation='relu', kernel_initializer='random_uniform'),
#     keras.layers.Dense(1024, activation='relu', kernel_initializer='random_uniform'),
#     # keras.layers.Dropout(0.5),
#     keras.layers.Dense(1024, activation='relu', kernel_initializer='random_uniform'),
#     keras.layers.Dense(512, activation='relu'),
#     keras.layers.Dense(num_of_signatures, activation='softmax')
# ])
#
# model.summary()
#
# model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=75)
#
# # testing using the test dataset
# _, test_accuracy_1 = model.evaluate(x_test, y_test)
#
# model.save('model.h5')

# neural network testing
model = keras.models.load_model('model.h5')

print('Seen values:')

# user 35
user = np.argmax(model.predict([[0.4248562844913726, 0.5860976937314559, 0.04837482288941347, 0.3389917446103405,
                                 0.08837255781386537, 0.010602273550635876, 0.014659182281609966, 2.040971586971126e-06,
                                 0.02114427860696517, 0.9806138933764136, 0.12522875816993465, 0.12012675244862685,
                                 0.40856465336867065, 0.0909360906263604, 0.061838893409275834, 0.5294117647058824]]))
print('result:', user)

# user 23
user = np.argmax(model.predict([[0.4368917414777884, 0.5079138519619878, 0.4069562359151474, 0.4316469595301688,
                                 0.2582529503212839, 0.041313288780718715, 0.059244417752919866, 1.1891475662889219e-05,
                                 0.0708955223880597, 0.9838449111470112, 0.12901960784313726, 0.603226425965047,
                                 0.5041149393220812, 0.3495821661425047, 0.13274439149134026, 0.7527010804321729]]))
print('result:', user)

# user 14
user = np.argmax(model.predict([[0.6973787208722501, 0.629417066744955, 0.4004781194450056, 0.10339772101179444,
                                 0.2500163432272329, 0.3627398256900762, 0.11840283910338682, 0.04898051607388792,
                                 0.3893034825870647, 0.9919224555735056, 0.17104072398190046, 0.5482043403111196,
                                 0.2316920072534524, 0.6628478566056011, 0.7206788329652447, 0.6338535414165666]]))
print('result:', user)

print('Unseen values:')

# user 38
user = np.argmax(model.predict([[0.1989879192862913, 0.7888499898189109, 0.02727500475503479, 0.17592617074788194,
                                 0.1723202796033483, 0.05486739971475013, 0.0874844509208092, 0.00017631402444134655,
                                 0.22885572139303484, 0.9806138933764136, 0.21254901960784314, 0.2901862876896485,
                                 0.35067652392244386, 0.2517321416340936, 0.4074160176682553, 0.9351740696278512]]))
print('result:', user)

# user 17
user = np.argmax(model.predict([[0.4873537224387776, 0.5859077170634134, 0.5612598449246609, 0.19867787146246565,
                                 0.23477223058896976, 0.03778916599818629, 0.03853594529066058, 2.410775255185253e-05,
                                 0.2562189054726368, 0.9806138933764136, 0.2240522875816993, 0.7954676397157673,
                                 0.3665783233365881, 0.6317943645994903, 0.29582703708008834, 0.8559423769507803]]))
print('result:', user)

# user 11
user = np.argmax(model.predict([[0.5100885786492371, 0.3956765505534707, 0.5053619918181225, 0.05657028933797016,
                                 0.0760772802158957, 0.01725357153467374, 0.0070710906191899375, 5.0448153385520325e-06,
                                 0.12686567164179105, 0.9822294022617124, 0.1565266106442577, 0.5453236028423276,
                                 0.20114381364206999, 0.7384462372186081, 0.2434034639079391, 0.3817527010804322]]))
print('result:', user)

# user 4
user = np.argmax(model.predict([[0.3605779758545407, 0.3594965945753029, 0.3452890860037785, 0.25345940750715223,
                                 0.2252390936121097, 0.03432019846853545, 0.03946985207912036, 7.473980562197342e-06,
                                 0.17288557213930347, 0.9757673667205171, 0.3003921568627451, 0.4561167658920683,
                                 0.3382619612219277, 0.39894587277583693, 0.27002208531907473, 0.5858343337334934]]))
print('result:', user)

# TRAIN TEST SPLIT
train2_count = 40 * split_percentage
test2_count = 40 * (1 - split_percentage)

train2_df = feature_vector[40 * user: int(40 * (user + 1) - test2_count)]
test2_df = feature_vector[int(40 * (user + 1) - test2_count):(40 * (user + 1)) - 1]

x_train2 = train2_df[
    ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
     'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]
y_train2 = train2_df['F']

x_test2 = test2_df[
    ['avgX', 'avgY', 'avgSDX', 'avgSDY', 'avgV', 'avgA', 'avgSDV', 'avgSDA', 'pen_down', 'pen_up', 'pen_ratio',
     'sign_width', 'sign_height', 'width_height_ratio', 'total_sign_duration', 'range_pressure']]
y_test2 = test2_df['F']

# # NEURAL NETWORK 2
# model = keras.Sequential([
#     keras.layers.Dense(256, input_shape=[x_train.shape[1]]),
#     # keras.layers.Dropout(0.5),
#     # keras.layers.Dense(2048, activation='relu', kernel_initializer='random_uniform'),
#     # keras.layers.Dense(2048, activation='relu', kernel_initializer='random_uniform'),
#     # keras.layers.Dropout(0.5),
#     keras.layers.Dense(256, activation='relu', kernel_initializer='random_uniform'),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
#
# model.summary()
#
# model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
#
# model.fit(x_train2, y_train2, epochs=100)
#
# # testing using the test dataset
# _, test_accuracy_2 = model.evaluate(x_test2, y_test2)
#
# model.save('model2.h5')
#
# print('The test accuracy of the model 1 is: ', test_accuracy_1)
# print('The test accuracy of the model 2 is: ', test_accuracy_2)
#
# neural network 2 testing
model2 = keras.models.load_model('model2.h5')

forgery = model2.predict([[0.49832421381066794, 0.7030368255464058, 0.5942152011250385, 0.3108114053487339,
                           0.17850419514959193, 0.016477817183470363, 0.023044989386517535, 4.931717077255626e-06,
                           0.19651741293532338, 0.975767366720517, 0.3202614379084967, 0.6804301901286729,
                           0.4068907797461292, 0.49106563722739943, 0.19900034871556435, 0.7563025210084033]])
print('result 2:', int(round(forgery[0][0])))
