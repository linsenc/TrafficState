# Traffic State Estimation Classification Problem


This project is part of the last chapter of my Ph.D. research, where the goal is to classify historial data into a number of traffic scenarios. Traffic scenarios could be off peak traffic demand scenarios of peak traffic demand scenarios. 

Data are collected from road sensors. For different traffic scenarios, such as different time of days, traffic behave differently. Thus, data from the same time of day (e.g., even peak hours) accross different days (e.g., one month) could be grouped into the same class, where the class label could be the index of the traffic scenario. Therefore, when data are collected at different time of days, they form several classes. Therefore, the goal of this problem is to create classification models to tell given a data point, which traffic scenario (i.e., class) should it belong to. This becomes a multi-class classification problem.

Detailed background information is included in my thesis, where we combine model-driven (i.e., mathematical models) and machine-learning algorithms here. This repository includes the machine learning algorithm used in this research. Also, a more detailed report is included in this repository. This readme file includes the key steps of this work.

# Data

Data include the link travel time of a number of road sections. A road section is defined as the section of a road between two intersections. In this work, we have 63 roads, therefore, the dimension of feature space is 63. We have 350 data observations from seven groups. Data dimension is 350*63.

# Models
We used three machine learning models, decision trees, random forest and k nearest neighbors. The reason of using decision trees and random forest is because we expect a non-linear relationship between features and estimators. For k nearest neighbors, we believe it makes "transportation sense": since traffic from different scenarios (an off peak scenarion and a peak scenario) should behavior differently, data from the same class should have close distance, and vice versa.

# Algorithm step
## Proprocessing

