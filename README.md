# DataMining
## Mining and Evaluating a Titanic-derived Dataset
### Youngjin Noh

### Structured abstract
Background: A number of studies have been conducted on survival probabilities through the dataset of the Titanic. This study describes the design of KNIME workflows designed to predict passenger survival through the Titanic dataset.

Objective: The appropriate method and its processes in classification problems are described.

Methods: Through 5-fold cross-validation, it was pre-processed and set to suit the K-Nearest-Neighbour and Random Forest method.

Results: The accuracy of random forests is 86.3%; it was selected as the ‘Information Gain Ratio’ in the Split Criterion.

Conclusion: Random forests get higher accuracy than other methods. Women are more likely to survive than men.
Keywords: Titanic, Random Forest, K-Nearest-Neighbour, Classification

### Introduction
The dataset of passengers age, gender, passenger ratings, and fares at the time of the Titanic sinking has been used by a number of researchers to determine the correlation between survival probabilities (Singh et al., 2017; Shetty et al., 2018; Sherlock et al., 2018; Singh et al., 2020). Even now, the dataset is actively being studied on a website Kaggle.com and correlations of various survival probabilities have been found (Singh et al., 2017). As such, new information and knowledge could be found at the stage of data mining and analysis (Maneth & Poulovassilis, 2017).
This study describes the design of KNIME workflows that implement data mining algorithms to predict survival of passengers in the Titanic disaster. Random Forest (RF) is selected for this classification operation. This study begins with a review of the data mining theory of other studies as a part of the literature review and will be introduced to the discussion of the most appropriate evaluation methods. The following sections describe the dataset and data pre-processing. This will explain the design of experiments and workflows and the process of finding optimal models. Finally, the performance of the model will be critically assessed, and some suggestions for improving the model are revealed in the study.
The primary purpose of this study is to evaluate the performance and accuracy of different algorithms and present appropriate data mining methods. In particular, studies are discussed critically and also mention limitations.

### Data mining theory
#### Random Forest Predictor (Singh et al., 2017)
The key and most commonly used data analysis method in this study is RF Predictor. RF algorithms could further improve the accuracy of classification models, which could be characterised by the survival of the Titanic disaster. This algorithm is an ensemble model that combines several decision trees (Singh et al., 2017; Tu & Chung, 1992). Regardless of the type of Target Column (categorical or numerical), it could be analysed and has the advantage of being able to grasp the relative importance of Attribute Selection. It could also prevent over-conformities occurring in a decision tree. However, if there are many variables, accuracy may be reduced, and non-continuous treatment may lead to errors in the prediction (Tu & Chung, 1992; Tin, 1995). According to a study by Singh et al. (2017), it was made using all the variables in the dataset. They claimed that Sex and Pclass played the most critical roles in survival probabilities and that he had the same results as the logistic regression algorithm he had previously studied. For example, they used the confusion matrix, and out of a total of 418 predictions, 384 accurate predictions were made to record 91.8% accuracy.

#### K-Nearest Neighbours (KNN) (Singh et al., 2020)
KNN method was also used in this study. KNN belongs to the most straightforward algorithm and classifies unknown datasets according to the similarity of adjacent results. When used as a classification, the result is determined by the majority vote of the nearest neighbour (Altman, 1992). In a study by Singh et al. (2020), the value of K was set to 3, and 83.95% accuracy was announced with 232 precise predictions. However, if the value of K is changed, the result may also be changed, and KNN has disadvantages when the class distribution is distorted. Moreover, the interpretation may vary depending on the opinion of researchers (Coomans & Massart, 1982).

#### Logistic Regression (Singh et al., 2017; Shetty et al., 2018; Singh et al., 2020)
Logistic regression is a way of revealing how Feature affects categorical response (binary) variables in determining the relationship between cause (description) and the result (response). The advantages of the above method are probability-based, which facilitates model implementation and explanation of how Feature affects Target. However, the disadvantages are that linearity and independence between features must be met (Tolles & Meurer, 2016). According to the research of Singh et al. (2020), the application of this logistic regression algorithm results in a probability of 80.24% with 214 predictions. They also argued that the chances of survival are high if female passengers board Queens and have more parents or children.

#### Discussion of an appropriate method for evaluation 
In this study, RF Algorithm should be used, and there are four reasons why it is appropriate. The first is that it would be the most adaptable and easy-to-use algorithm and It would be as simple to set up and use. Second, it would be a very accurate method. Some people would think that being easy and adaptable would mean sacrificing ability, but it would be not. Third, because it acts random sampling, RF is robust against overfitting. Finally, it is reasonably interpretable because relative feature importance can be obtained. It is not clear exactly how it made its decision, but generally, it could be seen which variables contributed more to the outcome than other variables (Breiman, 2001; Dietterich, 2000). However, a weakness of RF Model is that running could be slow. If the dataset is so large that it requires a large number of trees, it could be unrealistic or impractical because all trees have to vote on all observations even if it is evident that all processes must be carried out every time (Tu & Chung, 1992). In this study, however, it can be said that the Titanic dataset is relatively small and straightforward, making it very suitable for RF models.
In addition to the methods mentioned above, a number of researchers analysed the Titanic disaster dataset by methods of Naïve Bayes (Singh et al., 2017), Decision Tree (Singh et al., 2017; Shetty et al., 2018; Singh et al., 2020), Decision Tree with Hypertuning and Support Vector Machines (Shetty et al., 2018; Singh et al., 2020).

### Data exploration and preparation
The data is divided into two files, containing 1269 and 1243 rows in passenger table and ticket table respectively. Consist of attributes such as passengerlD, survived, ticket number, passenger name, sex, age, number of siblings, number of parents or children, salary, job title, cabin, ticket number, fare and embarked place. Data is in CSV file format. The titanic_ticket_data.csv data and titanic_personal_data.csv data consist of variables, as shown in Table 1 and 2 below. Therefore, it is necessary to find commonality between personal data and ticket data and to combine them using a node called ‘Joiner’. The raw dataset has incomplete or missing values that require pre-processing. There are 242 missing in age, 101 missing in salary and job, 934 missing values in cabin. Name, job, and ticket number of string features are deleted through 'Column Filter', and cabin is also excluded from the study due to its high missing values. To handle missing values in a dataset, only the selected variables (Fare, Embarked, Age, Salary) are extracted from the entered table using a node ‘Column Filter’ to output the table, and the missing values of the dataset are then processed by a node ‘Missing Value’ (Bakos, 2013; Meinl, 2012). The missing values in age, fare and salary are replaced by median and missing in Embarked are impute with unknown.

![Figure1](https://raw.githubusercontent.com/myaqueenas/DataMining/e98d5f39c87696451551d2e2246a17f42b97e8d6/Figures/Figure%202.PNG)

Figure 1. Handling Missing Value

It is also necessary to convert sex to numbers. For instance, 0 is allocated to men and 1 to women. Through ‘Statistic’ node, it can be examined the Statistics Table, which shows the figures of statistics, the Nominal Histogram Table, which shows the distribution of categorical variables by category, and the occurrences Table, which shows the percentage of the categories, frequencies, and frequencies of each variable. Furthermore, the histogram (Figure 1) below shows survived. The node ‘RowID’ in Input data allows the ‘RowID’ to be modified, replaced, or added as a variable (Bakos, 2013; Meinl, 2012).

![Figure2](https://raw.githubusercontent.com/myaqueenas/DataMining/e4a10022e7eac1ced117038af9684d7e2d79bd51/Table%201.PNG)

![Figure3](https://raw.githubusercontent.com/myaqueenas/DataMining/e4a10022e7eac1ced117038af9684d7e2d79bd51/Table%202.PNG)

![Figure4](https://raw.githubusercontent.com/myaqueenas/DataMining/e98d5f39c87696451551d2e2246a17f42b97e8d6/Figures/Histogram.png)

Figure 2. Survival Histogram

Having complex features could confuse some models in choosing the most informative attributes for partitioning on each node. Therefore, using a nodes 'Auto-Binner', 'Cell Replacer' and 'Table Creator' implement discretization of numerical attributes such as age, salary, and fare according to the quantitation range to replace numbers with labels. In other words, the age group is divided into Young, Young Adult, Adult and Senior, and Salary is divided into Poor, Working Class, Middle Class and Upper Class. Finally, Fare is divided equally into four stages: Steerage, Third Class, Second Class and First Class.

![Figure5](https://raw.githubusercontent.com/myaqueenas/DataMining/e98d5f39c87696451551d2e2246a17f42b97e8d6/Figures/Figure%203.PNG)

Figure 3. Discretization

Figure 4 illustrates that One-hot encoding is performed using the node 'One to Many' to convert each nominal value to a dummy variable (0 or 1). For classification method, ‘Number To String’ node means converting the target label ‘survived’ to a string value. Moreover, the ‘Partitioning’ node divides the input table into two output tables (training set and test set). In this study, it was divided into learning data and evaluation data to build a predictive model (Bakos, 2013; Meinl, 2012). The number of rows to output to the first partition is set at 80% per cent and the remaining 20%, the training set and test set contain 963 rows and remaining 241 rows, respectively.

![Figure6](https://raw.githubusercontent.com/myaqueenas/DataMining/e98d5f39c87696451551d2e2246a17f42b97e8d6/Figures/Figure%204.PNG)

Figure 4. Fuse tables and create test train split

### Experimental setup
Applying each simple condition, such as ‘Only Women Survive’ or ‘Only men and Not Young =>Died’, to the 'Rule Engine' node can achieve high accuracy without applying a complex model. In addition, 'Score' nodes are used to evaluate the performance of the model in a confusion matrix.

![Figure7](https://raw.githubusercontent.com/myaqueenas/DataMining/e98d5f39c87696451551d2e2246a17f42b97e8d6/Figures/Figure%205.PNG)

Figure 5. Simple Rules

Since training data is insufficient in the study, cross-validation of training is carried out on the training set for parameter optimization, and reliable evaluation of the model is carried out before applying the model to the test set. Use an 'X-Partitioner' node to start the loop, set the number of validations to 5, and select a random sampling method. Through an 'X-Aggregator' node, it is used to collect results from the predictor node and terminate the loop. Also, in this study, KNN's number of neighbours to consider was set at 3.

![Figure8](https://raw.githubusercontent.com/myaqueenas/DataMining/e98d5f39c87696451551d2e2246a17f42b97e8d6/Figures/Figure%206.PNG)

Figure 6. Comparing KNN and RF

‘Random Forest Learner’ Node performs model learning after setting up its target column and attributes for model learning. 'Random Forest Predictors' node uses a learned RF model to predict the target column. Through this node, the predicted results of the target column are added as new variables. Figure 7 reveals the performance of optimization for parameters in RF.

![Figure9](https://raw.githubusercontent.com/myaqueenas/DataMining/e98d5f39c87696451551d2e2246a17f42b97e8d6/Figures/Figure%207.PNG)

Figure 7. Tree Optimisation

### Results
Before applying complex models and methods, simple rule predictions were implemented. 86.085% of the target labels were correctly classified in training set through the prediction that 'only women survive'. This figure is higher than 61.682% and 76.324% respectively for the accuracy of 'Prediction Everyone dies' and 'Only men and not young die'. In addition, based on the evaluation of the training set, the prediction of "only women survive" was applied to the test set to record 85.477% accuracy. While the above results suggest that women are absolutely more likely to survive than men, variables affecting survival may also exist other than gender.
To compare the performance of the two methods, RF and KNN, these two methods are used separately to perform A 5-fold cross-validation for the training set. The comparison showed that RF (86.293% accuracy) has a lower error rate than the KNN (80.685% accuracy).
RF is further optimized through 5-fold cross-validation. In the first 'Random Forest Learner' node, it was selected as the 'Information Gain Ratio' in the Split Criterion and was set to the number of 250 decision tree models, resulting in 85.67% accuracy. On the other hand, the second 'Random Forest Learner' used 'Information Gain' as a Split Criterion, and with 500 Number of models, an accuracy 85.774% is obtained. The two accuracy figures are similar.
Based on cross-validation results, 500 models and 'Information Gain Ratio' are set and applied to the test set. As a result, the model has 85.892% accuracy in predicting survival in the test set. However, given the disproportionate case of survival in the dataset, the confusion matrix could be biased toward evaluation, so by comparing true positive rates with false positive rates, ROC metrics could address the effects of flag imbalances in instances. The AUC for the model was 0.8865, which means that the model is suitable in terms of performance evaluation of this model. In addition, the model performance of the shuffled data was 0.4998 AUC.

![Figure10](https://raw.githubusercontent.com/myaqueenas/DataMining/e98d5f39c87696451551d2e2246a17f42b97e8d6/Figures/Figure%208.PNG)

Figure 8. ROC curve

### Conclusion and reflections
When the Split Criterion was set to 'Information Gain Ratio', the conclusion was drawn in a relatively more appropriate way for RF. In conclusion, The RF Model has several advantages, but there are apparent disadvantages that large datasets are not suitable (Breiman, 2001; Tu & Chung, 1992). However, it can be judged to be very suitable for the Titanic dataset. Studies show that women and high-class people who pay high fares for Titanic maritime accidents are more likely to survive, but this is difficult to generalize in all maritime accidents and disasters (Singh et al., 2020). For higher accuracy, further research may use an adaptive boosting algorithm which is vulnerable to noisy data and outliers. However, it is less vulnerable to overfitting than to other learning algorithms (Rojas, 2009).

### Limitation
The Titanic dataset may be small to interpret the data and establish maritime accident knowledge. A lot of samples and data are needed to become academic theories and knowledge (Malterud et al., 2016). A completely different pattern could be found compared to other disasters or modern maritime accidents (Potter & Bolls, 2012; Long & Wall, 2013). 

### References
Altman, N. (1992). An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression. The American Statistician, 46(3), 175-185.

Bakos, G. (2013). KNIME essentials. Birmingham: Packt Publishing.

Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

Coomans, D., & Massart, D. (1982). Alternative k-nearest neighbour rules in supervised pattern recognition : Part 1. k-Nearest neighbour classification by using alternative voting rules. Analytica Chimica Acta, 136(C), 15-27.

Dietterich, T. (2000). An Experimental Comparison of Three Methods for Constructing Ensembles of Decision Trees: Bagging, Boosting, and Randomization. Machine Learning, 40(2), 139-157.

Long, P., & Wall, T. (2013). Media studies : texts, production, context (2nd ed.). Routledge.

Malterud, K., Siersma, V., & Guassora, A. (2016). Sample Size in Qualitative Interview Studies. Qualitative Health Research, 26(13), 1753-1760. doi: 10.1177/1049732315617444

Maneth, S., & Poulovassilis, A. (2017). Data science. Computer Journal, 60(3), 285-286.

Meinl, T. (2012). What's new in KNIME? Journal of Cheminformatics, 4(Suppl 1), Journal of Cheminformatics, 2012, Vol.4(Suppl 1)

Potter, R., & Bolls, P. (2012). Psychophysiological measurement and meaning cognitive and emotional processing of media . Routledge.

Rojas, R. (2009). AdaBoost and the super bowl of classifiers a tutorial introduction to adaptive boosting. Freie University, Berlin, Tech. Rep.

Sherlock, J., Muniswamaiah, M., Clarke, L., & Cicoria, S. (2018). Classification of Titanic Passenger Data and Chances of Surviving the Disaster. ArXiv.org, ArXiv.org, Oct 22, 2018.

Shetty, J., Pallavi, & Ramyashree. (2018). Predicting the Survival Rate of Titanic Disaster Using Machine Learning Approaches. 2018 4th International Conference for Convergence in Technology (I2CT), 1-5.

Singh, A., Saraswat, S., & Faujdar, N. (2017). Analyzing Titanic disaster using machine learning algorithms. 2017 International Conference on Computing, Communication and Automation (ICCCA), 2017, 406-411.

Singh, K., Nagpal, R., & Sehgal, R. (2020). Exploratory Data Analysis and Machine Learning on Titanic Disaster Dataset. 2020 10th International Conference on Cloud Computing, Data Science & Engineering (Confluence), 320-326.

Tolles, J., & Meurer, W. (2016). Logistic Regression: Relating Patient Characteristics to Outcomes. JAMA, 316(5), 533-534.

Tu, P., & Chung, J. (1992). A new decision-tree classification algorithm for machine learning. Proceedings Fourth International Conference on Tools with Artificial Intelligence TAI '92, 1992, 370-377.
