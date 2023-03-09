# Project-4

# Wine Quality Analysis Using Machine Learning

![wine-white-red](https://user-images.githubusercontent.com/112193116/222508329-c64a8002-01ac-4ebf-9c4a-bc16ded196c8.jpg)

Team Members: AJ Domingo, Samuel Heaton, Alice Johnson, 
Jananee Arjunan, Mia Tsivitse, William Julius

**PROJECT SCOPE**

Many factors are considered by wine experts and reviewers when determining the quality of a wine, mainly the taste and subjective experience of drinking the wine. Though rating wine can be subjective, there are factors that contribute to the final taste, aroma, and quality of the wine, including the geographic region in which the grapes were grown and the wine was produced, weather factors, and elevation. 

In this project, we explored data from wine reviews and supplemented this data with additional aggregated weather and elevation data. Using machine learning, our goal was to create a model that could predict the final rating of the wine as an indicator of its quality and potential sentiment amongst wine reviewers. 

# DATA ETL

Initial Data source: [Wine Reviews Dataset (Kaggle)](www.google.com)

After dropping data points with insufficient location information, we settled upon using only wines that were produced in the United States, as the US data points had sufficient location data. This data was then supplemented with a few different API calls: 
Google Places API was used to query the Winery name and Region in order to get latitude and longitude. 
Visual Crossing’s History Summary API endpoint was used to query the coordinates and return summary weather data from the last year. We were able to add data points for Max Temp, Min Temp, Precipitation, Humidity, and Heat Index. We decided on using these particular weather values because our research indicated that these all can have an effect on grape quality. 
Open Meteo API was used to query the coordinates and return an elevation datapoint. 

![API_screenshots](https://user-images.githubusercontent.com/112193116/223889653-e8c8f059-057f-4c73-b9c4-c39dc53a8f12.png)

Using sqlalchemy, the final dataset was loaded into an Amazon Web Services RDS Postgres SQL database for ease of use in extracting the data while working in a cloud environment for the modeling steps. 

# DATA EDA + VISUALIZATIONS

We also performed exploratory data analysis in jupyter notebook to find basic information about the dataset such as the average rating (points) amongst all wines, the average wine price by state, average wine rating (points) by state, average wine price, count of wines by state, etc.

Using the final dataset, we created visualizations in Tableau to aid our understanding and analysis. This included various bar and pie charts to conceptualize the dataset. We also created a heatmap and scatterplots of the correlation between our variables. Last, we created a chart highlighting the point distribution between positive, negative and neutral sentiment within the dataset. 

![eda_new](https://user-images.githubusercontent.com/112193116/223891151-18c32e9f-7b15-4571-9fb4-b2c3a9cd0ea1.png)

# TABLEAU NOTEBOOKS

[Tableau Workbook #1](https://public.tableau.com/app/profile/mia.tsivitse/viz/Wine_Quality_EDA_1/WineQualityEDA)

[Tableau Notebook #2](https://public.tableau.com/app/profile/william.julius/viz/winepointsandprice/Story9?publish=yes)


# DATA PRE-PROCESSING

The preprocessing was completed in Google Colab in the same notebook for our model so that we could easily use tensorflow. The data was retrieved from our RDS SQL Database using sqlalchemy in Google Colab and loaded into a pandas dataframe. The following steps were then taken to preprocess our data.

   We utilized SentimentAnalyzer and NaiveBayesClassifier from Natural Language Toolkit (NLTK) to create numerical data around sentiment to the description of each wine in our dataset, 
   
![nltk_new](https://user-images.githubusercontent.com/112193116/223891152-fa024dc6-201b-4791-bada-5cc7de4352e7.png)

![sentiment_new](https://user-images.githubusercontent.com/112193116/223891155-8ac51392-7499-4f45-a882-d02ab439fa57.png)
   
  * Dropped unnecessary columns: "wine_id", "country", “winery_name”, "description", "designation","taster_name", "taster_twitter_handle", "title". 
  * Created bins for the prices. 
  * Created bins for the points into 2 target values: 0 for below 90 points and 1 for above 90 points. 
  * Binned some varieties into “other” in order to contain outliers. 
  * Used pd.dummies to create dummies for our categorical/non-numerical features. 
  * Split the data into testing and training data. 
  * Scaled the data. 
  
 ![preprocessing_1](https://user-images.githubusercontent.com/112193116/223891154-6653adc8-7300-47d9-b29f-0b8dd8261506.png)
 
 **Target Values**
 
 ![target_new](https://user-images.githubusercontent.com/112193116/223891158-d3eeb8b3-ba8c-4dcd-bebc-cb6bddb46637.png)
 
 # DATA MODEL IMPLEMENTATION
 
 We decided to build a neural network model utilizing tensorflow. We built our model as follows: 

  * Input features - 247 (length of our X_train after getting dummies)
  * 2 hidden layers
  * 400 nodes - layer 1
  * 200 nodes - layer 2
  * Relu-activation function for the hidden layers
  * Trained over 100 epochs
  
![nnmodel_new](https://user-images.githubusercontent.com/112193116/223891319-f1cd9a7b-5716-435b-a56c-33a9a40d2598.png)

After running our model, we got a testing accuracy of 71 percent. We also noticed that our model was overfitting, and so we moved on to finding ways to optimize our model. 

![Screenshot 2023-03-08 201914](https://user-images.githubusercontent.com/112193116/223890398-ff09021e-bdab-44a9-ac71-0bb7671f1155.png)


**Random Forest Classifier**

Using RandomForestClassifier, we were able to achieve a model testing accuracy of 76 percent, but it was still over-fitting as the training accuracy was 99 percent.
<link>

# DATA MODEL OPTIMIZATION

Our main problem with our model was that it was over-fitting. We also tried using a MinMaxScaler instead of StandardScaler, but this also decreased the accuracy. 

**1. PCA**

At first, we tried using PCA to reduce the dimensions of our data, which resulted in 3 features and then applied that to our neural network model. This decreased the overall accuracy significantly so we abandoned that approach. 

<link>

**2. Feature importance used to look for best features using SelectFromModel**

<link>

We tried checking with different number of features and nodes combination.

**Top 20 features**

Initially, we tried using the top 20 features with 1 hidden layer and 50 Epochs to train the model using NN. We did not see any overfitting occur and the accuracy was 72 percent.

<links>

**Top 45 features**
<link>

**Top 65 features**
<link>

**Top 85 features**
<link>

After increasing the number of features and layers, the model was overfitting and the accuracy was dropping. Without overfitting the model, we achieved 72% accuracy by optimizing.

**3. MinMaxScaler**

We tried to use MinMaxScaler to try to improve accuracy, but it gave us a lower result at 53%.

<link>

**4. KerasTuner**

We utilized KerasTuner as an optimization to help us identify an optimal set of hyperparameters. We started with trying different max and step values and ran the KerasTuner model with 60 epochs. After testing 500/20, 400/20, and 500/30 max/step ratios, we found the highest accuracy to be 73% at 400/20. 











