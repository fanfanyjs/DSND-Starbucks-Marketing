# DSND-Starbucks-Marketing

## Description
This project is the capstone assignment of the Udacity Data Science Nanodegree. We are given stimulated data from Starbucks reward mobile app that tracks the users who have received, viewed and used the offers, as well as any other transactions they have made. Our task is to find out what offer types elicit the most response from which type of customers.

In particular, we are given three datasets:


The data has included ten different offer types. There are bogo (buy-one-get-one), discount, and informational offers, with a variety of distribution channels, minimum spend required and amount of reward. There can be three actions associated with a single offer - receive, view and complete. The offer can be marked as completed even when a user hasn't viewed the offer before using it.

## Dependencies

Python 3.5 is used to create this project and the following libraries are used:

- Machine Learning Libraries: NumPy, Pandas, Sciki-Learn, XGBoost
- Data Visualisation: Matplotlib, Seaborn
- Python Serialization: Joblib

## Methodology

In this exercise, I have assumed each offer received by a customer to be independent of others received by the same customer. I have built a classification model to predict whether each offer sent to each customer, would successfully prompt the user to make a purchase. This assumption is made as each customer is usually sent multiple types of offers of different offers (not all types) with different sequences. As a result, there is no like for like comparison on a customer level as each receives different treatment. 

I have experimented with Logistics, SVC, Random Forest and Gradient Boosting models using the following features:
- demographics (age, income, gender)
- time since joined
- type of offer received (one-hot encoded)
- interaction variables between demographics and type of offer received (The interaction variables are specifically tested for non-tree based techniques such as Logistics and SVC. Tree-based techniques like Random Forest and Gradient Boosting have inherently considered interaction between variables due to its sequential selection of variables.

## Results and Discussion

## Credits
Thanks to Udacity and Starbucks for preparing the data and structuring this piece of assignment.

Several papers and blogs that I have come across have also inspired my approach towards this problem.  
