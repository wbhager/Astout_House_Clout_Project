print("hello, world!2")

print("Making this not green")

print("This is a change!")

# Importing pandas package so I can use dataframes and all the features that come with manipulating them
# as well as being able to read different kinds of raw data in
import pandas as pd
import numpy as np



# Displaying all the columns so that it's easier to see
pd.set_option('display.max_columns', None)




real_estate_data_1 = pd.read_csv("real_estate_data.csv")


real_estate_data_2 = pd.read_excel("Homes For Sale and Real Estate.xlsx")


real_estate_data_3 = pd.read_csv("Housing_Price_Data.csv")


all_real_estate = pd.concat([real_estate_data_1, real_estate_data_2, real_estate_data_3])
print(all_real_estate.tail(1))


# Creating the Total_Price variable from the combination of all of the price and Price variables. They never have the same value.
all_real_estate["Total_Price"] = all_real_estate["price"].combine_first(all_real_estate["Price"])


# Creating the Total_Beds variable from the combination of all of the beds variables. They never have the same value.
all_real_estate["Total_Beds"] = all_real_estate["bedrooms"].combine_first(all_real_estate["Beds"])


# Creating the Total_Baths variable from the combo of all the baths variables. They never have the same value.
all_real_estate["Total_Baths"] = all_real_estate["bathrooms"].combine_first(all_real_estate["Bath"])


# Creating the Total_Sq_Feet_Living variable from the combo of all the sqfeet variables. They never have the same value.
all_real_estate["Total_Sq_Feet_Living"] = all_real_estate["sqft_living"].combine_first(all_real_estate["Sq.Ft"])


# Creating the Total_Lot_Area variable from the combo of all the lot area variables. They never have the same value.
all_real_estate["Total_Lot_Area"] = all_real_estate["sqft_lot"].combine_first(all_real_estate["area"])


# Creating the Total_Floors variable from the combo of all the floors variables. They never have the same value.
all_real_estate["Total_Floors"] = all_real_estate["floors"].combine_first(all_real_estate["stories"])


# Creating the Basement? variable that can tell if a house that has at least 1 square foot of basement or not
all_real_estate["Basement?"] = np.where(all_real_estate["sqft_basement"] > 0, "yes", "no")


# Combining Basement variables
all_real_estate["Basement"] = all_real_estate["basement"].combine_first(all_real_estate["Basement?"])


# Creating the Price_per_Sq_Foot variable
all_real_estate["Price_per_Sq_foot"] = all_real_estate["Total_Price"] / all_real_estate["Total_Sq_Feet_Living"]


# Creating the Confirmed_Extra_Features variable containing the number of confirmed extra features 
all_real_estate["Confirmed_Extra_Features"] = ((all_real_estate["mainroad"] == "yes").astype(int)) + \
                                              ((all_real_estate["hotwaterheating"] == "yes").astype(int)) + \
                                              ((all_real_estate["airconditioning"] == "yes").astype(int)) + \
                                              (all_real_estate["prefarea"].astype(int)) + \
                                              (all_real_estate["parking"]) + \
                                              np.select(
                                                   [all_real_estate["furnishingstatus"] == "furnished",
                                                    all_real_estate["furnishingstatus"] == "semi-furnished",
                                                    all_real_estate["furnishingstatus"] == "unfurnished"],
                                                    [2, 1, 0],
                                                    default = 0
                                               )


# Dropping the original vars
all_real_estate = all_real_estate.drop(["price", "Price", "bedrooms", "Beds", "bathrooms", "Bath", "sqft_living", "Sq.Ft", "sqft_lot", "area", "floors", "stories", "basement", "Basement?"], axis = 1)






# Getting all the columns in variable form so that I can put the newly combined columns (which happen to be
# the important ones) in the dataset at the very beginning
columns_to_be_moved = ["Total_Price", "Total_Beds", "Total_Baths", "Total_Sq_Feet_Living", "Total_Lot_Area", "Total_Floors"]

# Columns not in the list of columns
remaining_columns = [col for col in all_real_estate.columns if col not in columns_to_be_moved]

# New Column order
new_column_order = columns_to_be_moved + remaining_columns

# Reordering dataframe
all_real_estate = all_real_estate[new_column_order]

print(all_real_estate.head())





# Performing linear regression to fill in the missing values for all of the missing Total_Sq_Feet_Living values

# Dropping rows with NaN to avoid errors in regression
all_clean = all_real_estate.dropna(subset = ["Total_Price", "Total_Lot_Area"])

# Separating data into training and prediction sets (Data with Total_Sq_Feet_Living and then the data without Total_Sq_Feet_Living)
training_data = all_clean.dropna(subset = ["Total_Sq_Feet_Living"])
predict_data = all_clean[all_clean["Total_Sq_Feet_Living"].isna()]

# Preparing the training data by turning it into a numpy array
X_train = training_data[["Total_Price", "Total_Lot_Area"]].values
y_train = training_data[["Total_Sq_Feet_Living"]].values

# Adding a column for intercept data
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Preparing the prediction data
X_predict = predict_data[["Total_Price", "Total_Lot_Area"]].values
X_predict = np.hstack([np.ones((X_predict.shape[0], 1)), X_predict])

# Fitting the linear regression model
coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predicting the missing values
predicted_values = X_predict @ coefficients

# Imputing the missing values for Total_Sq_Feet_Living
all_real_estate.loc[all_real_estate["Total_Sq_Feet_Living"].isna(), "Total_Sq_Feet_Living"] = predicted_values






# Doing it all again for the total lot area!

# Changing the all_clean dataset to only include the observations without NA values for both Total_Price and Total_Sq_Feet_Living
all_clean = all_real_estate.dropna(subset = ["Total_Price", "Total_Sq_Feet_Living"])

# Rewriting the training data to be all of the observations where Total_Lot_Area has a valid value and rewriting prediction data to be the ones without
training_data = all_clean.dropna(subset = ["Total_Lot_Area"])
predict_data = all_clean[all_clean["Total_Lot_Area"].isna()]

# Prepping training data by turning it into a numpy array
X_train = training_data[["Total_Price", "Total_Sq_Feet_Living"]].values
y_train = training_data[["Total_Lot_Area"]].values

# Adding a column of ones to the left of the xtrain data to act as the intercepts
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Prepping the prediction data and adding intercepts
X_predict = predict_data[["Total_Price", "Total_Sq_Feet_Living"]].values
X_predict = np.hstack([np.ones((X_predict.shape[0], 1)), X_predict])

# Fitting the linear regression model
coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predicting the missing values
predicted_values = X_predict @ coefficients

# Imputing the missing values for Total_Lot_Area
all_real_estate.loc[all_real_estate["Total_Lot_Area"].isna(), "Total_Lot_Area"] = predicted_values



# My current dataset
print(all_real_estate)

# Getting all the counts of Nas across every column
nan_counts = all_real_estate.isna().sum()
print(nan_counts)




# Variables to add: number of features, housing grade, 

# Variables to add if I need more and have no more ideas: number of people recommended to live there, 


"""
Features:

- Main Road, Hot Water Heating, Each level of parking (1 garage is 1, 2 garage is 2, etc),
Preferred area, furnishing status (1 for semi, 2 for full)
"""






# At the end of this code (the data transformation, this code is just in this file for now), turn it into a csv using DataFrame.to_csv().
# In the graphs file, read the CSV, and then do all the graphing stuff. And then when running the python files in the end, run the transformation first.
