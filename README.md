# NLP - Is the job posting Fake or Not?
## The Data:
- The dataset is from Kaggle: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
- Narrowed the scope to only job postings on the US market
- 10656 total observations, 740 fake and 9926 real
- A class imbalance of Minority class: 6.85% and Majority class: 93.15%
- 16 initial total features (4 text features, job 'description' being the text to be evaluated with NLP techniques)
- Target variable: 'fraudulent' 0 is real, 1 is fake

## Data Cleaning and Initial EDA
Most of the features were categorical and missing many values.  The only columns not missing values were the job 'title' (dropped as a result of over 1000 different non-standardized titles), 'location', and job 'description'.  The remaining categorical features with large missing values were evaluated with a Chi-Squared test to determine whether or not an imputed value was statistically significant in relation to the posting being fake or real.
