print("Hi! Welcome to my cleaning and transformation python file.")

print("We doing stuff!")

print("We doing stuff more!")

print("more stuff?")

# Importing pandas package so I can use dataframes and all the features that come with manipulating them
# as well as being able to read different kinds of raw data in
import pandas as pd

real_estate_data_1 = pd.read_csv("real_estate_data.csv")

print(real_estate_data_1.head())

real_estate_data_2 = pd.read_excel("Homes For Sale and Real Estate.xlsx")

print(real_estate_data_2.head())

real_estate_data_3 = pd.read_csv("Housing_Price_Data.csv")

print(real_estate_data_3.head())  







# At the end of this code, turn it into a csv using DataFrame.to_csv()