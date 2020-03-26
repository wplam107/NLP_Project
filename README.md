# NLP - Is the job posting Fake or Not?
## The Data:
- The dataset is from Kaggle: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
- Original data set can be found: http://emscad.samos.aegean.gr
- Original data set contains 17,881 worldwide job postings
- Narrowed the scope to only job postings for the US market
- 10656 total observations, 740 fake and 9926 real
- A class imbalance of Minority class: 6.85% and Majority class: 93.15%
- 16 initial total features (4 text features, job 'description' being the text to be evaluated with NLP techniques)
- Target variable: 'fraudulent' 0 is real, 1 is fake

## Data Cleaning and Initial EDA
Most of the features were categorical and missing many values.  The only columns not missing values were the job 'title' (dropped as a result of over 1000 different non-standardized titles), 'location' (dummied by state), and job 'description' (text document to be converted to a mean word vector value).  The remaining categorical features with large missing values were evaluated (dummied or dropped) with a Chi-Squared test to determine whether or not an imputed value was statistically significant in relation to the posting being fake or real.

## Baseline Models
4 baseline models were used: Logistic Regression, RandomForest Classifier, Support Vector Classifier, XGBoost Classifier.  The baseline models were run without the text data (job 'description') as a feature.  The lack of feature interactions were expected to bias the data against the Logistic Regression model.  All models were run inconjunction with PCA to reduce the feature space.  


# NLP Project

1. Created a classification model that used text data features from job description postings in the United States to determine if a job posting was fraudulent or not. 
2. Performed Exploratory Data Analysis to identify key traits of job descriptions which are fraudulent in nature.

Presentation Slides: 




