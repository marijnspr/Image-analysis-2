import os

train_data_dir = "C:/Users/marij/Documents/Image-analysis-foto/Fruit/train"
test_data_dir = "C:/Users/marij/Documents/Image-analysis-foto/Fruit/test"

# Print all files in the training directory
for root, dirs, files in os.walk(train_data_dir):
    for file in files:
        print(os.path.join(root, file).replace('\\', '/'))

# Print all files in the training directory
for root, dirs, files in os.walk(test_data_dir):
    for file in files:
        print(os.path.join(root, file).replace('\\', '/'))
