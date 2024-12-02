# Code for Evolutionary Algorithms for SKU Optimization
Eliza Knapp and Sidd Tiwari

### File Structure
All relevant code to generate images in the paper is in the FinalCode directory.
- algorithm.py is a class which carries out one experiment start to finish and also graphs the results.
- convert_and_forecast.ipynb takes the original product_sales.csv from Kaggle and converts it into ForecastedInformation.csv by forecasting the length of the policy optimization period in advance.
- results.ipynb uses the Algorithm class to create all the results that are in section 4, Results, of our paper.
- tune_pop_size.ipynb uses a modified Algorithm class to show how we decided the upper bound population size in our hyperparameter search. Its figures appear in section 3.3, Hyperparameter Search, of our paper.
