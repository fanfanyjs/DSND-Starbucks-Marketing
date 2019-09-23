# DSND-Starbucks-Marketing

## Motivation
This project is the capstone assignment of the Udacity Data Science Nanodegree. We are given stimulated data from Starbucks reward mobile app that tracks the users who have received, viewed and used the offers, as well as any other transactions they have made. Our task is to find out what offer types elicit the most response from which type of customers.

In particular, I would like to investigate the below questions:
- Which particular groups of people are likely to complete offers in general?
- Which particular offers are likelier to evoke response or completion in general?
- Which offer should we tailor to a specific customer?

## Data
We are given three datasets:
- Profile: customer data with id and demographic details
- Portfolio: list of offers with distribution channel, minimum spend, reward and effective duration
- Transcript: list of events (offer receipt, view, and redemption) occurence time and transaction amount

The data has included ten different offer types. There are bogo (buy-one-get-one), discount, and informational offers, with a variety of distribution channels, minimum spend required and amount of reward. There can be three actions associated with a single offer - receive, view and complete. The offer can be marked as completed even when a user hasn't viewed the offer before using it.

## Dependencies

Python 3.5 is used to create this project and the following libraries are used:

- Machine Learning Libraries: math, NumPy, Pandas, Sciki-Learn, XGBoost
- Data Visualisation: Matplotlib, Seaborn
- Python Serialization: joblib
- Date: date
- Warnings: warning

Customised functions are also written using these libraries and are imported from the following python file:
- customised_fct: RawPreprocess, TreatmentModelling, EDAplot

## Methodology

A hybrid of classification model and regression models are run to predict the expected net revenue uplift of a customer. This model will then be used to recommend a specific offer to a customer, if a customer id, or a list of customer details is provided.

1. A classification model will be fit onto the data to predict whether a customer would complete an offer. Features for prediction would include the type of offer received, the customer demographics and behavioural metrics and interaction between the two.
2. Regression models will be fit onto the data with success flag. One regression model will be fitted for purchases associated with each offer as distribution of purchase amount varies amongst offers.
3. A single regression model will be fit onto purchase data that is not associated with any offers.

The logic is as follows:

P(success) = P(viewed) * P(completed)

Net Revenue Uplift(per order) = Revenue with offer - Difficulty - Revenue without Offer

Expected Uplift(per order) = P(success) * Net Revenue Uplift


I have experimented with XGBoost, Random Forest and Linear models (Logistics for classification, Linear Regression and Ridge for regression) using the following features:
- demographics (age, income, gender)
- time since joined
- type of offer received (one-hot encoded)
- interaction variables between demographics and type of offer received (The interaction variables are specifically tested for non-tree based techniques such as Logistics for classification. Tree-based techniques like Random Forest and Gradient Boosting have inherently considered interaction between variables due to its sequential selection of variables.

In this exercise, I have assumed each offer received by a customer to be independent of others received by the same customer. This assumption is made as each customer is usually sent multiple types of offers of different offers (not all types) with different sequences. As a result, there is no like for like comparison on a customer level as each receives different treatment. 

## Results and Discussion
**Who to target in the offer:** As seen from the EDA section, those who earn high income (>70000) and female spend much more than minimum spend, as compared to most males and those who earn lower income (<35000), who are likely to just spend just above the minimum spend. Therefore, the first two group should be targeted if Starbucks would like to increase its profits.

**What offer to push:** In general, 'fafd' and '2298' discount offers have the highest redemption rate and expected net revenue (after deducting reward) uplift. Around 61% and 57% of those who receive the respective offers are likely to make purchases. And they are likely to pay up to 5 and 3 dollars more than they would otherwise do if they don't receive an offer.

**Modelling and Customised Offer Recommendation:** A hybrid of classifier and regression models are used to predict the probability of a person redeeming offer and the potential revenue uplift should they make a purchase.

XGBoost performs the best for all these models as it could predict values whose distribution are skewed and have regularised functions. Its classifier yields a precision of 0.71 and recall of 0.58. Its limited accuracy is likely because of imbalance of its success classes, and insufficient data to give more insight in purchase behaviours prior to offers being distributed. To mitigate this, balancing techniques like sub-sampling, over-sampling or SMOTE can be used. More transaction information prior to the offer distribution could also be extracted.

Its regressors have an R2 of 21% to 71% (most of them are above 50%). By removing outlier values (people who pay a lot of money per purchase), the error has significantly reduced. Nevertheless, the regression models don't perform as ideal for some of the offers as some of the distribution are bimodel. Gaussian mixture models could potentially be of better fit in this situation. Also, there might be insufficient prior transaction data to help predict a customer's purchase frequency and behaviour.

My Medium post on this analysis can be accessed [here](https://medium.com/@yap.fantasy/how-to-target-promotional-offers-in-starbucks-to-increase-roi-1b801eb9a4b5).

## Credits
Thanks to Udacity and Starbucks for preparing the data and structuring this piece of assignment.

Several papers and blogs that I have come across have also inspired my approach towards this problem.  
[1] Gordon, B. R., Zettelmeyer, F., Bhargava, N., & Chapsky, D. (2017). A Comparison of Approaches to Advertising Measurement: Evidence from Big Field Experiments at Facebook. SSRN Electronic Journal. doi: 10.2139/ssrn.3033144
[2] Gelman, A., & Hill, J. (n.d.). Causal inference using regression on the treatment variable. Data Analysis Using Regression and Multilevel/Hierarchical Models, 167–198. doi: 10.1017/cbo9780511790942.012
