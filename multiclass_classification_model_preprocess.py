import math
import numpy as np
import pandas as pd

file_name = 'u1s1'

list_df = []
list_size = []

# COMBINING ALL THE FILES INTO ONE LIST
count = 0
for i in range(1, 41):
    for j in range(1, 41):
        file_name = 'signature_data/U' + str(i) + 'S' + str(j) + '.txt'

        # Reading into a dataframe and appending it to a list
        df = pd.read_csv(file_name, delimiter=' ', names=['X', 'Y', 'TS', 'T', 'AZ', 'AL', 'P'], header=None,
                         skiprows=1)
        list_df.append(df)

        # Creating list_size of storing number of rows in each file
        rows, cols = df.shape
        list_size.append(rows)

# CONVERTING INTO A LARGE DATAFRAME
# ARRAY FORMAT
df_array = np.vstack(list_df)

# CREATING A DATAFRAME
final_dataset = pd.DataFrame(df_array)
final_dataset.columns = ['X', 'Y', 'TS', 'T', 'AZ', 'AL', 'P']
print('Final dataset(top 5 values): ')
print(final_dataset.head())

# ATTRIBUTES CALCULATION
forgery = []

no_of_pts = []
velocity_array = []
x_velocity_array = []
y_velocity_array = []
avg_pressure = []
avg_x = []
avg_y = []
pen_up = []
pen_down = []
pen_ratio = []
std_dev_x = []
std_dev_y = []
acceleration_array = []
avg_vel = []
avg_acc = []
std_dev_vel = []
std_dev_acc = []
sign_height = []
sign_width = []
width_height_ratio = []
pts_width_ratio = []
max_pressure = []
var_pressure = []
range_pressure = []

distance = 0
index = 0
count = 0
x_distance = 0
y_distance = 0
pressure = 0

temp = 0
time = 0

tempX = []
tempY = []
tempP = []

for x in list_size:
    for i in range(x):
        j = i - 1
        if j >= 0:
            distance += ((final_dataset['X'][count + j] - final_dataset['X'][count + i]) ** 2 + (
                    final_dataset['Y'][count + j] - final_dataset['Y'][count + i]) ** 2) ** 0.5

            x_distance += abs(final_dataset['X'][count + j] - final_dataset['X'][count + i])

            y_distance += abs(final_dataset['Y'][count + j] - final_dataset['Y'][count + i])

            time = final_dataset['TS'][count + x - 1] - final_dataset['TS'][count]

            pressure += final_dataset['P'][count + i]

    tempX = final_dataset['X'][count:count + x]
    tempY = final_dataset['Y'][count:count + x]
    tempP = final_dataset['P'][count:count + x]

    count += x

    # no_of_pts
    no_of_pts.append(x)

    # pen-down
    pen_down.append(final_dataset['T'][count:count + x].sum())

    # pen_up
    pen_up.append(x - final_dataset['T'][count:count + x].sum())

    # pen_ratio
    pen_ratio.append(float(final_dataset['T'][count:count + x].sum() / (x - final_dataset['T'][count:count + x].sum())))

    # velocity calculation
    velocity_array.append(float(distance / time))

    # x_velocity calculation
    x_velocity_array.append(float(x_distance / time))

    # y_velocity calculation
    y_velocity_array.append(float(y_distance / time))

    # avg_pressure calculation
    avg_pressure.append(float(pressure / x))

    # avg_x calculation
    avg_x.append(x_distance / x)

    # avg_y calculation
    avg_y.append(y_distance / x)

    # std_dev_x calculation
    std_dev_x.append(math.sqrt((x_distance - (x_distance / x)) ** 2 / x))

    # std_dev_y calculation
    std_dev_y.append(math.sqrt((y_distance - (y_distance / x)) ** 2 / x))

    # acceleration calculation
    acceleration_array.append(float(distance / (time * time)))

    # avg_vel caalculation
    avg_vel.append(float(float(distance / time) / x))

    # avg_acc calculation
    avg_acc.append(float(float(distance / (time * time)) / x))

    # std_dev_vel calculation
    std_dev_vel.append((float(distance / time) - float(float(distance / time) / x)) / x)

    # std_dev_acc calculation
    std_dev_acc.append((float(distance / (time * time)) - float(float(distance / (time * time)) / x)) / x)

    # sign_width
    sign_width.append(max(tempX) - min(tempX))

    # sign_height
    sign_height.append(max(tempY) - min(tempY))

    # width_height-ratio
    width_height_ratio.append(float((max(tempX) - min(tempX)) / (max(tempY) - min(tempY))))

    # pts_width_ratio
    pts_width_ratio.append(float(x / (max(tempX) - min(tempX))))

    # max_pressure
    max_pressure.append(max(tempP))

    # std_dev_pressure
    var_pressure.append(float((pressure - float(pressure / x)) ** 2) / x)

    # range_pressure
    range_pressure.append(max(tempP) - min(tempP))

    # initialization
    distance = 0
    x_distance = 0
    y_distance = 0
    time = 0
    pressure = 0

print('no_of_pts:', no_of_pts[:5], len(no_of_pts))

print('velocity     :', velocity_array[:5], len(velocity_array))

print('x_velocity   :', x_velocity_array[:5], len(x_velocity_array))

print('y_velocity   :', y_velocity_array[:5], len(y_velocity_array))

print('avg_pressure :', avg_pressure[:5], len(avg_pressure))

print('avg_x :', avg_x[:5], len(avg_x))

print('avg_y :', avg_y[:5], len(avg_y))

print('pen-up :', pen_up[:5], len(pen_up))

print('pen-down :', pen_down[:5], len(pen_down))

print('pen_ratio:', pen_ratio[:5], len(pen_ratio))

print('std_dev_x :', std_dev_x[:5], len(std_dev_x))

print('std_dev_y :', std_dev_y[:5], len(std_dev_y))

print('acceleration :', acceleration_array[:5], len(acceleration_array))

print('avg_velocity :', avg_vel[:5], len(avg_vel))

print('avg_acceleration :', avg_acc[:5], len(avg_acc))

print('std_dev_vel :', std_dev_vel[:5], len(std_dev_vel))

print('std-dev_acc :', std_dev_acc[:5], len(std_dev_acc))

print('sign_width:', sign_width[:5], len(sign_width))

print('sign_height:', sign_height[:5], len(sign_height))

print('width_height_ratio:', width_height_ratio[:5], len(width_height_ratio))

print('pts_width_ratio:', pts_width_ratio[:5], len(pts_width_ratio))

print('max_pressure:', max_pressure[:5], len(max_pressure))

print('var_pressure:', var_pressure[:5], len(var_pressure))

print('range_pressure:', range_pressure[:5], len(range_pressure))

i = 1
forgery = []
forgery_nums = []

# LABEL GENERATION - ONE FOR EACH USER
while i <= len(velocity_array):
    for j in range(40):
        current_user = (i//40)
        forgery.append('U' + str(current_user))
        forgery_nums.append(current_user)
    i += 40

# # LABELS GENERATION - TWO FOR EACH USER
# cs = 0
# while i <= len(velocity_array):
#     for j in range(20):
#         current_user = ((i + j) // 40) + 1
#         status_g = 'U' + str(current_user) + 'G'
#         forgery.append(status_g)
#         forgery_nums.append(cs)
#         i += 1
#     cs += 1
#     for k in range(20):
#         current_user = ((i + 20 + k) // 40)
#         status_f = 'U' + str(current_user) + 'F'
#         forgery.append(status_f)
#         forgery_nums.append(cs)
#         i += 1
#     cs += 1

# COMBINE IT INTO A FEATURE VECTOR

fv_dictionary = {'no_of_pts': no_of_pts, 'velocity': velocity_array, 'x_velocity': x_velocity_array,
                 'y_velocity': y_velocity_array,
                 'avg_pressure': avg_pressure, 'avg_x': avg_x, 'avg_y': avg_y, 'pen_up': pen_up, 'pen_down': pen_down,
                 'pen_ratio': pen_ratio, 'std_dev_x': std_dev_x, 'std_dev_y': std_dev_y,
                 'acceleration': acceleration_array,
                 'avg_velocity': avg_vel, 'avg_acceleration': avg_acc, 'std_dev_vel': std_dev_vel,
                 'std_dev_acc': std_dev_acc,
                 'sign_width': sign_width, 'sign_height': sign_height, 'width_height_ratio': width_height_ratio,
                 'pts_width_ratio': pts_width_ratio, 'max_pressure': max_pressure, 'var_pressure': var_pressure,
                 'range_pressure': range_pressure}

feature_vector = pd.DataFrame(fv_dictionary)

# NORMALIZE THE FEATURE VECTOR
feature_vector = ((feature_vector - feature_vector.min()) * 1000 / (feature_vector.max() - feature_vector.min()))

feature_vector['forgery'] = forgery_nums

print('Normalized feature vector(top 5 values): ')
print(feature_vector.head(50))
feature_vector.to_csv('feature_vector_normalized.csv', index=False)
