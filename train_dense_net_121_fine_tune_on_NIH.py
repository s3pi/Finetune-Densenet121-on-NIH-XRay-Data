import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from sklearn.utils import shuffle
from models import dense_net_imagenet_base_model_non_trainable

def name_numbers(length, number):
    return '0' * (length - len(str(number))) + str(number)

def data_loader():
    img_path = []
    labels = []
    for i in range(1, 13):
        name = name_numbers(3, i)
        current_path = data_path + "/images_" + str(name)
        for file in os.listdir(current_path):
            img_path.append(os.path.join(current_path, file))
            labels.append(i-1)
    img_path = np.asarray(img_path)
    labels = np.asarray(labels)
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(img_path, labels, test_size = 0.2)

    return X_train_paths, X_test_paths, y_train, y_test

def open_metric_files():
    per_batch_metrics_file = open(result_files_path + '/training_per_batch_metrics' + '.txt', 'a')
    per_epoch_metrics_file = open(result_files_path + '/training_per_epoch_metrics' + '.txt', 'a')
    test_metrics_file = open(result_files_path + '/test_metrics' + '.txt', 'a')

    return per_batch_metrics_file, per_epoch_metrics_file, test_metrics_file

def write_metric_files(files, values):
    for i in range(len(files)):
        files[i].write(str(values[i])+ '\n')

def close_metric_files(files):
    for each_file in files:
        each_file.close()

def save_models(prev_epoch, e):
    if prev_epoch == -1:
        model.save_weights(model_weights_path + "/model_last_but_1.h5")
    elif e == 1:
        model.save_weights(model_weights_path + "/model_last.h5")
    elif (e // save_every == 0) or (e == num_epochs - 1):
        os.rename(model_weights_path + "/model_last.h5", model_weights_path + "/model_last_but_1.h5")
        model.save_weights(model_weights_path + "/model_last.h5")

def load_from_paths(X_test_paths):
    X_test = np.zeros((len(X_test_paths), size, size, 3))
    for i in range(len(X_test_paths)):
        X_test[i] = cv2.imread(X_test_paths[i])

    return X_test

def evaluate(X_test, y_test):
    num_of_batches = int(len(X_test_paths)/batch_size)

    for batch_num in range(num_of_batches):
        batch_X_train = np.zeros((batch_size, size, size, 3))
        batch_y_train = np.zeros((batch_size, 1))
        b = 0

        for j in range(batch_num*batch_size, min((batch_num+1)*batch_size, len(X_test_paths))):
            batch_X_train[b, :, :] = cv2.imread(X_test_paths[j]) / 255.0        
            batch_y_train[b] = y_test[j]
            b += 1

        loss, accuracy = model.test_on_batch(batch_X_train, batch_y_train)
        # print('epoch_num: %d, batch_num: %d, loss: %f, accuracy: %f\n' % (e, batch_num, loss, accuracy))
        # write_metric_files([per_batch_metrics_file], [[e, batch_num, loss, accuracy]])

        per_epoch_test_loss += loss
        per_epoch_test_acc += accuracy

    per_epoch_test_loss = per_epoch_test_loss / num_of_batches
    per_epoch_test_acc = per_epoch_test_acc / num_of_batches
    write_metric_files([test_metrics_file], [[e, per_epoch_test_loss, per_epoch_test_acc]])

    if per_epoch_test_loss < min_loss:
        model.save_weights(model_weights_path + '/step1_' + str(count))
        count+=1
        if count >= 10:
            os.remove('step1_'+str(count-10)+'.h5')
        prev_epoch = 0
        min_loss = per_epoch_test_loss

def train():
    X_train_paths, X_test_paths, y_train, y_test = data_loader()
    # X_test = load_from_paths(X_test_paths)
    count = 0
    prev_epoch = -1
    for e in range(num_epochs):
        X_train_paths, y_train = shuffle(X_train_paths, y_train, random_state = 2)
        per_batch_metrics_file, per_epoch_metrics_file, test_metrics_file = open_metric_files()
        per_epoch_loss = 0.0
        per_epoch_accuracy = 0.0
        per_epoch_test_loss = 0.0
        per_epoch_test_acc = 0.0
        min_loss = 100.0
        num_of_batches = int(len(X_train_paths)/batch_size)

        for batch_num in range(num_of_batches):
            batch_X_train = np.zeros((batch_size, size, size, 3))
            batch_y_train = np.zeros((batch_size, 1))
            b = 0

            for j in range(batch_num*batch_size, min((batch_num+1)*batch_size, len(X_train_paths))):
                batch_X_train[b, :, :] = cv2.imread(X_train_paths[j]) / 255.0        
                batch_y_train[b] = y_train[j]
                b += 1

            loss, accuracy = model.train_on_batch(batch_X_train, batch_y_train)
            print('epoch_num: %d, batch_num: %d, loss: %f, accuracy: %f\n' % (e, batch_num, loss, accuracy))
            write_metric_files([per_batch_metrics_file], [[e, batch_num, loss, accuracy]])

            per_epoch_loss += loss
            per_epoch_accuracy += accuracy

        per_epoch_loss = per_epoch_loss / num_of_batches
        per_epoch_accuracy = per_epoch_accuracy / num_of_batches
        write_metric_files([per_epoch_metrics_file], [[e, per_epoch_loss, per_epoch_accuracy]])

        evaluate(X_test, y_test)
        
        close_metric_files([per_batch_metrics_file, per_epoch_metrics_file, test_metrics_file])

###############################################################################################
data_path = "/home/ada/Preethi/XRay_Report_Generation/Data/ChestXray-NIHCC/Images"
result_files_path = "/home/ada/Preethi/XRay_Report_Generation/Code/Results"
model_weights_path = "/home/ada/Preethi/XRay_Report_Generation/Code/Model_Weights"
model = dense_net_imagenet_base_model_non_trainable() # For the first few epochs, train only the FC layer and keep the base_model non trainable
batch_size = 30
num_epochs = 10000
# save_every = 1
size = 1024
###############################################################################################
train()

