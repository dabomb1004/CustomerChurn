# Customer Churn

Using a dataset that contains Customer Information at a bank, we tried to predict whether or not the customer will leave using a neural network from the python sklearn package. The dataset is as follows: 

(variable columns with example data type) 
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000

# Preprocessing

To preprocess the data, we took out irrelevant data such as customer id and surname and we saved the relevant info into variable x and the target variable into y. Pandas was used to split the columns into X and y. 

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

Next, we had to encode our categorical (country column) and binary data (male/female or gender column). 

We imported LabelEncoder and OneHotEncoder from sklearn.preprocessing and ColumnTransformer from sklearn.compose. Since Geography column has text, we need to encode them to numbers. If we use LabelEncoder to convert the text into numbers we will get a new transformed column that looks as follows: 

Geography       ----->     Geography(Transformed Column)

Spain                      0
France                     1
Germany                    2



But since the Geography column values (France, Spain, Germany) are categorical and have no relation to one another, this won't do. Currently the model will interpret the different numbers in the newly transformed column as having some kind of order where 0 < 1 < 2. Since this is not the case, we need to do something that will avoid this model misinterpretation. What we need to use is the OneHotEncoder. If you would like to learn more about the OneHotEncoder, please check it out this excellent medium article by Sunny: https://contactsunny.medium.com/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621

# Splitting between Training and Testing. Then Normalize Data 

With our data correctly encoded, we now split the data between Training (80%) and Testing (20%) using sklearn's train_test_split() function. 

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)

After, we do one more preprocessing. We normalize the data using StandardScaler() from sklearn.preprocessing. We normalize the data to create better prediction accuracy and prevent extremely large or small values such as in Estimated Salary or Balance from affecting the model too heavily. 

# Creating the model 

Finally we use Keras in Python to create the ANN model. We use 3 hidden layers with 5 neurons with a relu activation function and make sure our input_dim equals the # of variable columns contained in our model (excluding the target column in this case the Exited Column). The output layers uses a sigmoid activation function as this is prediction a binary outcome (0 or 1 where 0 means the customer did not exit and 1 means the customer exited). 

For the compile stage we use binary_crossentropy as our loss and we use accuracy for our metrics. 

Note: I am using Google Colab to run this model. If you are on Colab you will need to change the notebook settings to use GPU. To do this go to Edit > Notebook Settings > And click GPU as Hardware Accelerator. 

Lastly, fit your classfier to the X_train and y_train datasets

# Last Steps

A confusion matrix was created to see the True Positive vs Fasle Positives and True Negative vs False Negatives. 

Then the root means squared error (rmse) was calculated and at the end, a prediction was made on a new set of data. The threshold was set at 0.5 and anything greater than that threshold would predict the customer would exit the bank. 

With 
