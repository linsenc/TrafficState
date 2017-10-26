# Traffic State Estimation Classification Problem


This project is part of the last chapter of my Ph.D. research, where the goal is to classify historial data into a number of traffic scenarios. Traffic scenarios could be off peak traffic demand scenarios of peak traffic demand scenarios. 

Data are collected from road sensors. For different traffic scenarios, such as different time of days, traffic behave differently. Thus, data from the same time of day (e.g., even peak hours) accross different days (e.g., one month) could be grouped into the same class, where the class label could be the index of the traffic scenario. Therefore, when data are collected at different time of days, they form several classes. Therefore, the goal of this problem is to create classification models to tell given a data point, which traffic scenario (i.e., class) should it belong to. This becomes a multi-class classification problem.

Detailed background information is included in my thesis, where we combine model-driven (i.e., mathematical models) and machine-learning algorithms here. This repository includes the machine learning algorithm used in this research. Also, a more detailed report is included in this repository. This readme file includes the key steps of this work.

# Data

Data include the link travel time of a number of road sections. A road section is defined as the section of a road between two intersections. In this work, we have 63 roads, therefore, the dimension of feature space is 63. 

Please note that the data are pre-cleaned in this research. In real world, the data munging part should take a significant amount of time and would be absolute key in machine learning model pipeline.

# Models
We use three machine learning models, decision trees, random forest and k nearest neighbors. The reason of using decision trees and random forest is because we expect a non-linear relationship between features and estimators. For k nearest neighbors, we believe it makes "transportation sense": since traffic from different scenarios (an off peak scenarion and a peak scenario) should behavior differently, data from the same class should have close distance, and vice versa.

Please note that we have tried other models, e.g., logitics regression and SVM. As it turns out, they perform inferior and not included here.  


# Algorithm steps
## Preprocessing and feature engineering

This step is crucial and should check several things including but not limited to:
--Data size
--Missing values (how to impute) and anomaly
--Correlations between features
--Multicolinearity 
--Interaction terms

Note that data size, non-numeric features, the distributions of class screening are not included in this step because the data we get can fit into my laptop, traffic data are all numeric, and samples of each class are roughly even. However, these screening steps should be done for general ML problems.

In this project, checking correlations between features is the key. This makes sense, because traffic conditions on adjacent roads are indeed correlated. This makes us think the features of several neighoring roads are indeed telling the same story: if congestion happens some region, the roads in the region are equally affected. In our case, feature engineering is the key and brings improvement in model performance. 

In addition, business-driven features are equally important. This will determine which features should (or should not) be included. For instance, if none of the models perform well, despite of parameters fine-turning, if could be the fact the key features are not included in the feature space. In this project, we have data of a limited number of roads, we do not know the data of all the roads of the cities of our interest. Therefore, we are better off with data of the full road network. Note that in my thesis, the sparse data issue does not pose a significant problem because we also use a mathematical traffic model which can help "guess" data from roads where data are missing.

We use feature diemnsion and normalization techniques to preprocess the features. For feature dimension reduction, we have tried 1) Linear discriminant analysis (LDA) and 2) Principle component analysis (PCA). Where PCA is an unsupervised way of feature reduction, LDA technique requires the knowledge of class label. LDA helps to find projection vectors that maximize the inter class variances while minimize the intra class varirances. In our case, LDA is critical in improving model performances.

For normalization, this work normalizes the reduced feature into bounded integers [0,M]. This is a traffic control problem specific normalization technique, in order to configuration traffic control hardware. Nonetheless, normalization is proved to be crucial to the K nearest neighbor algorithm, as the unit of features does matter to compute the distance. This noramlization technique is inspired by the work of Anderson (2015). 

Lastly, the parameters of LDA/PCA, i.e., the number of eigenvectors, and and the number M used by the normalziation step needed to be choosen by cross-validation. That is, the choice of LDA/PCA parameter and the choice of machine learning model parameters should be done simulteanously. i.e., We compare the performance of a machine learning model under the choice of LDA projection vectors, where we find the best machine learning model and the best LDA/PCA dimension together (we treat LDA dimension parameter as a parameter of a machine learning model) through the same cross-validation process. 

## Modeling building and cross validation
Try a number of nodels, and then work stardardized training and cross-validation process. The model evaluation metric we look at is the the evaluation metric is the misclassification rate. 

The cross-validation process is setup with the grid-search package of sklearn with the goal of finding a better parameter grid. In this work, we do exhaustive grid search because the computational requirement is not heavy. However, for large dataset, randomized parameter optimization is preferred. 

In the .py file, we showcase the model building and cross-validation using decision trees, random forest and KNN. 

## Ensemble 
The final step of this work to ensemble good models with corresponding parameters. The goal is to develop en ensemble of a diversity of good models. In this work, we use the majority vote of decision trees, random forest and KNN.

This is a brief discription of the data-drive machine learning models we've developed. I haven't listed all the technical details and results of the algorithms. The latter is included in the .pdf file.

