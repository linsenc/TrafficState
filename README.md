# Traffic State Estimation Classification Problem


This project is part of the last chapter of my Ph.D. research, where the goal is to classify historial data into a number of traffic scenarios. Therefore, the traffic control strategies developed in the past for these scenarios could be applied. This is helpful for online optimization problem in my research, because these pre-developed strategies could be used in real-time, or used as initial solutions to facilitate online optimization algoriths. The traffic scenarios of our interest could be off peak traffic demand patterns, peak hour traffic demand scenarios, etc. 

Data are collected from road sensors. For different traffic scenarios, such as different time of days, we expect traffic to behave differently. Nontheless, traffic behavior at the same peak hour from different days could be similar. Therefore, from the dataset, we would be to group data in several classes, where the class label could be the index of a prevalent traffic scenario. Thus, the goal of this problem is to create classification models to tell given a data point (an observation), which traffic scenario (i.e., class) should it belong to. This becomes a multi-class classification problem.

Detailed background information is included in my thesis, where we combine model-driven (i.e., mathematical models) and machine-learning algorithms. However, this repository only includes the machine learning algorithm used in this research. This readme file includes the key steps of this work. The detailed report is also included in my thesis.

## Data

The data include the link travel time of a number of road sections. A road section is defined as the section of a road between two intersections. In this work, we have 63 roads, therefore, the dimension of feature space is 63. 

The data are stored in sqlite database. We use SQL queries to get the road sensor data of our interest. An example of the query is included as an ipython notebook file in this repository.

Please note that the data are pre-cleaned in this research. In real world, the data munging part should take a significant amount of time and would be absolute key in machine learning model pipeline.

## Models
We use three machine learning models, decision trees, random forest and k nearest neighbors. The reason of using decision trees and random forest is because we expect a non-linear relationship between features and estimators. For k nearest neighbors, we believe it makes "transportation sense": since traffic from different scenarios (an off peak scenarion and a peak scenario) should behavior differently, data from the same class should have close distance, and vice versa.

Please note that we have tried other models, e.g., logistics regression and SVM. As it turned out, they perform inferior and are not included here.  


## Algorithm steps
### Preprocessing and feature engineering

This step is crucial and should check several things including but not limited to:

--Data size

--Missing values (how to impute) and anomaly

--Correlations between features

--Multicolinearity 

--Interaction terms

Note that data size, non-numeric features, the distributions of class screening are not included in this step because the data we get can fit into my laptop, traffic data are all numeric, and samples of each class are roughly even. However, these screening steps should be done for general ML problems.

In this project, checking correlations between features is the key. This makes sense, because traffic conditions on adjacent roads are indeed correlated. This makes us to think the that features of several neighoring roads should indeed telling the same story: if congestion happens in some region, the roads in the region are more or less equally affected. In our case, feature engineering is the key and brings improvement in model performance. 

In addition, business-driven features are equally important. This will determine which features should (or should not) be included. For instance, if none of the models perform well despite of parameters fine-turning, if could be the fact the key features are not included in the feature space. In this project, we have data of a limited number of roads, we do not know the data of all the roads of the cities of our interest. This can potentially cause machine learning models to fail to have a good inference of the network level traffic patterns. Note that in my thesis, the sparse data issue does not pose a significant problem because we also use a mathematical traffic model which can help "guess" data from roads where data are missing.

We use feature diemnsion and normalization techniques to preprocess the features. For feature dimension reduction, we have tried 1) Linear discriminant analysis (LDA) and 2) Principle component analysis (PCA). Where PCA is an unsupervised way of feature reduction, LDA technique requires the knowledge of class label. LDA helps to find projection vectors that maximize the inter class variances while minimize the intra class varirances. In our case, LDA is critical in improving model performances and outperform PCA.

For normalization, this work normalizes the reduced feature into bounded integers [0,M]. This is a traffic control problem specific normalization technique, in order to configurate traffic control hardware. Nonetheless, normalization is proved to be crucial to the K nearest neighbor algorithm, as the unit of features does matter to compute the distance. 

Lastly, the parameters of LDA/PCA, i.e., the number of eigenvectors, and and the number M used by the normalziation step needed to be choosen by cross-validation. That is, the choice of LDA/PCA parameter and the choice of machine learning model parameters should be done simulteanously. i.e., We compare the performance of a machine learning model under the choice of LDA projection vectors, where we find the best machine learning model and the best LDA/PCA dimension together (we treat LDA dimension parameter as a parameter of a machine learning model) through the same cross-validation process. 

### Modeling building and cross validation
This step tries a number of nodels, and then use stardardized training and cross-validation process. The model evaluation metric we look at is the misclassification rate. 

The cross-validation process is setup with the grid-search package of sklearn with the goal of finding a better parameter grid. In this work, we do exhaustive grid search because the computational requirement is not heavy. However, for large dataset, randomized parameter optimization is preferred. 

In the python file **researchDDModel.py**, we showcase the model building and cross-validation using decision trees, random forest and KNN. 

### Model Ensemble 
The final step of this work to ensemble good models with corresponding parameters. The goal is to develop en ensemble of a diversity of good models. In this work, we use the majority vote of decision trees, random forest and KNN.

## Final note
This is a brief discription of the data-drive machine learning models we've developed. I haven't listed all the technical details and results of the algorithms. Please feel free to ask me for any questions.

