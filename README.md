# Traffic State Estimation Classification Problem


This project is part of the last chapter of my Ph.D. research, where the goal is to classify historial data into a number of traffic scenarios. Traffic scenarios could be off peak traffic demand scenarios of peak traffic demand scenarios. 

Data are collected from road sensors. For different traffic scenarios, such as different time of days, traffic behave differently. Thus, data from the same time of day (e.g., even peak hours) accross different days (e.g., one month) could be grouped into the same class, where the class label could be the index of the traffic scenario. Therefore, when data are collected at different time of days, they form several classes. Therefore, the goal of this problem is to create classification models to tell given a data point, which traffic scenario (i.e., class) should it belong to. This becomes a multi-class classification problem.

Detailed background information is included in my thesis, where we combine model-driven (i.e., mathematical models) and machine-learning algorithms here. This repository includes the machine learning algorithm used in this research. Also, a more detailed report is included in this repository. This readme file includes the key steps of this work.

# Data

Data include the link travel time of a number of road sections. A road section is defined as the section of a road between two intersections. In this work, we have 63 roads, therefore, the dimension of feature space is 63. 

Please note that the data are pre-cleaned in this research. In real world, the data munging part should take a significant amount of time and would be absolute key in machine learning pipeline.

# Models
We used three machine learning models, decision trees, random forest and k nearest neighbors. The reason of using decision trees and random forest is because we expect a non-linear relationship between features and estimators. For k nearest neighbors, we believe it makes "transportation sense": since traffic from different scenarios (an off peak scenarion and a peak scenario) should behavior differently, data from the same class should have close distance, and vice versa.

Please note that we have tried other models, e.g., logitics regression and SVM. The perform inferior and not included here.  


# Algorithm steps
## Preprocessing and feature engineering

This step is crucial and should check several things including but not limited to:
--Data size
--Missing values and anomaly
--Correlations between features
--Multicolinearity 
--Interaction terms

Note that data size, non-numeric features, the distributions of class screening are not included in this step because the data we get can fit into my laptop, traffic data are all numeric, and samples of each class are roughly even. However, these screening steps should be done for general ML problems.

In this project, checking correlations between features is the key. This makes sense, because traffic conditions on adjacent roads are indeed correlated. This makes us think the features of several neighoring roads are indeed telling the same story: if congestion happens some region, the roads in the region are equally affected. In our case, feature correlation is the key and brings improvement in model performance. 

In addition, business-driven features are equally important. This will determine which features should (or should not) be included. For instance, if none of the models perform well, despite of parameters fine-turning, if could be the fact the key features are not included in the feature space. In this project, we have data of a limited number of roads, we do not know the data of all the roads of the cities of our interest. Therefore, we are better off with data of the full road network. Note that in my thesis, the sparse data issue does not pose a significant problem because we also use a mathematical traffic model which can help "guess" data from roads where data are missing.

1) Feature dimension reduction through 1) Linear discriminant analysis (LDA) and 2) Principle component analysis (PCA). LDA 


2) Normalization: this is critical to the K nearest neighbor algorithm, because the unit of . This is inspired by the work of Anderson (2015), which is a problem specific technique and has been used to address traffic signal control problems. 

Please note that for the sake of illustration, the choice of LDA/PCA parameters and normalization parameters should be done by cross-validation

## Modeling building and selection
1.Try a number of nodels,training and validation

4) Ensemble models






