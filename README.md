# Project-4

# Wine Quality Analysis Using Machine Learning

![wine-white-red](https://user-images.githubusercontent.com/112193116/222508329-c64a8002-01ac-4ebf-9c4a-bc16ded196c8.jpg)

Team Members: AJ Domingo, Samuel Heaton, Alice Johnson, 
Jananee Arjunan, Mia Tsivitse, William Julius

**PROJECT SCOPE**

Many factors are considered by wine experts and reviewers when determining the quality of a wine, mainly the taste and subjective experience of drinking the wine. Though rating wine can be subjective, there are factors that contribute to the final taste, aroma, and quality of the wine, including the geographic region in which the grapes were grown and the wine was produced, weather factors, and elevation. 

In this project, we explored data from wine reviews and supplemented this data with additional aggregated weather and elevation data. Using machine learning, our goal was to create a model that could predict the final rating of the wine as an indicator of its quality and potential sentiment amongst wine reviewers. 

**DATA ETL**

Initial Data source: [Wine Reviews Dataset (Kaggle)](www.google.com)

After dropping data points with insufficient location information, we settled upon using only wines that were produced in the United States, as the US data points had sufficient location data. This data was then supplemented with a few different API calls: 
Google Places API was used to query the Winery name and Region in order to get latitude and longitude. 
Visual Crossing’s History Summary API endpoint was used to query the coordinates and return summary weather data from the last year. We were able to add data points for Max Temp, Min Temp, Precipitation, Humidity, and Heat Index. We decided on using these particular weather values because our research indicated that these all can have an effect on grape quality. 
Open Meteo API was used to query the coordinates and return an elevation datapoint. 

<links>

Using sqlalchemy, the final dataset was loaded into an Amazon Web Services RDS Postgres SQL database for ease of use in extracting the data while working in a cloud environment for the modeling steps. 


**DATA EDA + VISUALIZATIONS**

We also performed exploratory data analysis in jupyter notebook to find basic information about the dataset such as the average rating (points) amongst all wines, the average wine price by state, average wine rating (points) by state, average wine price, count of wines by state, etc.

Using the final dataset, we created visualizations in Tableau to aid our understanding and analysis. This included various bar and pie charts to conceptualize the dataset. We also created a heatmap and scatterplots of the correlation between our variables. Last, we created a chart highlighting the point distribution between positive, negative and neutral sentiment within the dataset. 

<link>


**TABLEAU NOTEBOOKS**

[Tableau Workbook #1](https://public.tableau.com/app/profile/mia.tsivitse/viz/Wine_Quality_EDA_1/WineQualityEDA)

[Tableau Notebook #2](https://public.tableau.com/app/profile/william.julius/viz/winepointsandprice/Story9?publish=yes)


**DATA PRE-PROCESSING**

The preprocessing was completed in Google Colab in the same notebook for our model so that we could easily use tensorflow. The data was retrieved from our RDS SQL Database using sqlalchemy in Google Colab and loaded into a pandas dataframe. The following steps were then taken to preprocess our data.

   We utilized SentimentAnalyzer and NaiveBayesClassifier from Natural Language Toolkit (NLTK) to create numerical data around sentiment to the description of each wine in our dataset, 
   
   <link>
   
  * Dropped unnecessary columns: "wine_id", "country", “winery_name”, "description", "designation","taster_name", "taster_twitter_handle", "title". 
  * Created bins for the prices. 
  * Created bins for the points into 2 target values: 0 for below 90 points and 1 for above 90 points. 
  * Binned some varieties into “other” in order to contain outliers. 
  * Used pd.dummies to create dummies for our categorical/non-numerical features. 
  * Split the data into testing and training data. 
  * Scaled the data. 
  
 <link>
 
 **Target Values**
 
 <link>
 
 **DATA MODEL IMPLEMENTATION**
 <link>
 
 We decided to build a neural network model utilizing tensorflow. We built our model as follows: 

  * Input features - 247 (length of our X_train after getting dummies)
  * 2 hidden layers
  * 400 nodes - layer 1
  * 200 nodes - layer 2
  * Relu-activation function for the hidden layers
  * Trained over 100 epochs

After running our model, we got a testing accuracy of 71 percent. We also noticed that our model was overfitting, and so we moved on to finding ways to optimize our model. 






