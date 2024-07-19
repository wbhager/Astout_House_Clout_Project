("This is the temporary master branch")



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


# Subbing in 0 values for NaN for all my features since it is adding to NAs when adding my extra features
all_real_estate["mainroad"] = all_real_estate["mainroad"].fillna(0)
all_real_estate["hotwaterheating"] = all_real_estate["hotwaterheating"].fillna(0)
all_real_estate["airconditioning"] = all_real_estate["airconditioning"].fillna(0)
all_real_estate["prefarea"] = all_real_estate["prefarea"].fillna(0)
all_real_estate["parking"] = all_real_estate["parking"].fillna(0)
all_real_estate["waterfront"] = all_real_estate["waterfront"].fillna(0)
all_real_estate["furnishingstatus"] = all_real_estate["furnishingstatus"].fillna(0)





"""
Before you normalize, impute these values

airconditioning: 100k, 150k
hotwaterheating: 125k, 175k
furnishingstatus: 250k, 350k, 450k
parking: 400k, 550k
prefarea: 750k, 1M
waterfront: ?, ?

Waterfront note: Some cheap houses are waterfront, maybe make each house have a 5 percent chance to have it regardless of price? Maybe we can
do this with other variables too?


Create FULL_PACKAGE variable, if the house contains the max possible amount of features, put True, if not, False


"""

# Imputing values for my feature variables. If the house price reaches a certain threshold, the house may have a particular feature, and if it reaches a higher 
# threshold, it will almost certainly contain the feature.

# Function style 1 (50%, 99%)
airconditioning_thresholds = [100000, 150000]
hotwaterheating_thresholds = [125000, 175000]
prefarea_thresholds = [750000, 1000000]

# Function style 2 (50%, 99/50%, 99/99%)
furnishingstatus_thresholds = [250000, 350000, 450000]

# Function style 3 (50%, 99/50%, 99/99/50%, 99/99/99%)
parking_thresholds = [350000, 450000, 550000, 650000]

# Function style 4 (5% initially, 50%, 99%)
waterfront_thresholds = [1000000, 1500000]


def feature_determinant(df, list_of_thresholds, feature, initial_prob = 0.5):
     """
     df: dataframe with the real estate data I want to look at

     list_of_thresholds: List of integers. They are price thresholds, so if a price of a house crosses that
     threshold, then there becomes an updated chance that the particular hoouse contains that threshold. Houses
     that already contain the feature keep the feature.

     feature: String. If the name matches with a feature that exists within the all_real_estate and the 
     function decides for the feature to be added, that variable within all_real_estate gets appended to match
     what the function says to set as the new value.


     """
     for i in range(len(df)):
          price = df.at[i, "Total_Price"]
          feature_value = df.at[i, feature]

          if feature_value == 0:
               if price >= list_of_thresholds[-1]:
                    if np.random.rand() <= 0.99:
                         df.at[i, feature] = 1
               elif (len(list_of_thresholds) > 3) and (price >= list_of_thresholds[-2]):
                    if np.random.rand() <= 0.99:
                         df.at[i, feature] = 1
               elif (len(list_of_thresholds) > 2) and (price >= list_of_thresholds[-3]):
                    if np.random.rand() <= 0.99:
                         df.at[i, feature] = 1
               elif (len(list_of_thresholds) > 1) and (price >= list_of_thresholds[1]):
                    if np.random.rand() <= 0.99:
                         df.at[i, feature] = 1
               elif price >= list_of_thresholds[0]:
                    if np.random.rand() <= initial_prob:
                         df.at[i, feature] = 1
    

feature_determinant(all_real_estate, airconditioning_thresholds, 'airconditioning')
feature_determinant(all_real_estate, hotwaterheating_thresholds, 'hotwaterheating')
feature_determinant(all_real_estate, prefarea_thresholds, 'prefarea')
feature_determinant(all_real_estate, furnishingstatus_thresholds, 'furnishingstatus')
feature_determinant(all_real_estate, parking_thresholds, 'parking')
feature_determinant(all_real_estate, waterfront_thresholds, 'waterfront', initial_prob=0.05)




# Creating the Confirmed_Extra_Features variable containing the number of confirmed extra features 
all_real_estate["Confirmed_Extra_Features"] = ((all_real_estate["mainroad"] == "yes") & (all_real_estate['mainroad'].notna())).astype(int) + \
                                              ((all_real_estate["hotwaterheating"] == "yes") & (all_real_estate['hotwaterheating'].notna())).astype(int) + \
                                              ((all_real_estate["airconditioning"] == "yes") & (all_real_estate['airconditioning'].notna())).astype(int) + \
                                              ((all_real_estate["prefarea"] == "yes") & (all_real_estate['prefarea'].notna())).astype(int) + \
                                              (all_real_estate["parking"]) + \
                                              (all_real_estate["waterfront"]) + \
                                              np.select(
                                                   [all_real_estate["furnishingstatus"] == "furnished",
                                                    all_real_estate["furnishingstatus"] == "semi-furnished",
                                                    all_real_estate["furnishingstatus"] == "unfurnished"],
                                                    [2, 1, 0],
                                                    default = 0
                                               )


# Dropping the original vars
all_real_estate = all_real_estate.drop(["price", "Price", "bedrooms", "Beds", "bathrooms", "Bath", "sqft_living", "Sq.Ft", "sqft_lot", "area", "floors", "stories", "basement", "Basement?"], axis = 1)












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




# Creating the Price_per_Sq_Foot_Living variable
all_real_estate["Price_per_Sq_Foot_Living"] = all_real_estate["Total_Price"] / all_real_estate["Total_Sq_Feet_Living"]


# Creating the Price_per_Sq_Foot_Lot
all_real_estate["Price_per_Sq_Foot_Lot"] = all_real_estate["Total_Price"] / all_real_estate["Total_Lot_Area"]


# Creating the Housing_Grade variable
"""

My most complex variable!

First, we gotta update the features. Most of these houses obviously have them,
so we're just gonna add them. Sometimes.

hotwaterheating, airconditioning, parking, prefarea, and furnishingstatus will
be updated alongside price. If a price reaches a certain threshold, for a specified
variable, there will be a 50 percent chance the house has it. If the price reaches a much
higher threshold, the feature will be included in 99 percent of all the houses.

"""


all_real_estate["Basement_Normalized"] = all_real_estate["Basement"].map({"yes": 1, "no": 0})

all_real_estate["Total_Sq_Feet_Living_Normalized"] = 1 - (all_real_estate["Total_Sq_Feet_Living"] - all_real_estate["Total_Sq_Feet_Living"].min()) / (all_real_estate["Total_Sq_Feet_Living"] - all_real_estate["Total_Sq_Feet_Living"].max())



# Once you finish this part, mess around with the sorting and indexing so that you are more familiar with how to do it and find out
# more about the data from this






# Getting all the columns in variable form so that I can put the newly combined columns (which happen to be
# the important ones) in the dataset at the very beginning
columns_to_be_moved = ["Total_Price", "Total_Beds", "Total_Baths", "Total_Sq_Feet_Living", "Price_per_Sq_Foot_Living", "Total_Lot_Area",
                        "Price_per_Sq_Foot_Lot", "Basement", "Confirmed_Extra_Features", "Total_Floors"]

# Columns not in the list of columns
remaining_columns = [col for col in all_real_estate.columns if col not in columns_to_be_moved]

# New Column order
new_column_order = columns_to_be_moved + remaining_columns

# Reordering dataframe
all_real_estate = all_real_estate[new_column_order]

print(all_real_estate.head())



# My current dataset
print(all_real_estate.iloc[250:260])

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


