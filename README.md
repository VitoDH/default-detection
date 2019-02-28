

# Default Detection on P2P lending

#### **Author**: Dehai Liu

**Department of Mathematics, Sun Yat-Sen University**



## 1. Abstract

This project mainly focus on data mining in the P2P lending data. Based on LDA topic model, we can extract information from the loan statement given by the borrower. Combined with the traditional features (gender,age,target value,etc) in default detection, we are able to predict the probability of default with random forest, which demonstrates high accuracy and straightforward interpretation.



## 2. Data Description

`Ren Ren Dai`(https://www.renrendai.com/) is one of the largest online P2P lending platform in China.  Using the crawling software Octopus, I obtain around 10,000 records of online lending with the 20 features and 1 label (default or not default) :



**Lending Hard Information**

|   Attributes   |          Definitions           |
| :------------: | :----------------------------: |
| interest rate  | the interest rate of the loan  |
|  target value  |     the amount of the loan     |
| lending period | the period of holding the loan |



**Lending Soft Information**

|    Attributes    |                         Definitions                          |
| :--------------: | :----------------------------------------------------------: |
| repayment method | 1: pay interest first 2: pay interest and principal together |
|  property loan   |     1: short-term turnover 2: personal consumption ....      |
|  loan statement  |          the statement for loan before application           |



**Lending Personal Info**

|   Attributes    |                    Definitions                    |
| :-------------: | :-----------------------------------------------: |
|       age       |                continuous variable                |
|     gender      |            0: female, 1: male, 2: NULL            |
|    education    | 1: high school 2: junior college 3: undergrad ... |
|    marriage     |     1: divorce 2: married 3: single 4: widow      |
| census register |                 province in China                 |
|     income      |             1: <1000 2: 1000-2000 ...             |
| house property  |                   1: yes 0: no                    |
|   house loan    |                   1: yes 0: no                    |
|       car       |                   1: yes 0: no                    |
|    car_loan     |                   1: yes 0: no                    |
|  type company   |                  type of company                  |
|    industry     |                 1: IT 2: food ...                 |
|  scale company  |        (# of staffs) 1: <10 2: 10-100 ...         |
|    type job     |                    type of job                    |
|    workplace    |                the location of job                |
|    time job     |            1: <1 year 2: 1-3 years ...            |



## 3. Data Preprocessing

### (a) Load the data

* Load p2pData.csv

Note : the irrelevant feature in the raw data has been removed



### (b) Outlier Detection

Perform outlier detection on continuous variables **target value** and **age** with Tukey Mehod, i.e defining the data points out of 1.5 times the Interquartile range as outleir.



**Target Value**

*  Before removing the outlier

<img src="https://github.com/VitoDH/default-detection/raw/master/img/with_outlier_Target.png" style="zoom:60%" />

* After removing the outlier

<img src="https://github.com/VitoDH/default-detection/raw/master/img/without_outlier_Target.png" style="zoom:60%" />



**Age**

* Before removing the outlier

<img src="https://github.com/VitoDH/default-detection/raw/master/img/with_outlier_Age.png" style="zoom:60%" />

* After removing the outlier

<img src="https://github.com/VitoDH/default-detection/raw/master/img/without_outlier_Age.png" style="zoom:60%" />

The distribution of the features become less skewed after removing the outliers.



### (c) Missing Values

* For ordinal variables, impute the missing values with the median.
* For categorical variables without order, remove them directly since they are just a small portion of the samples.



### (d) Scaling

Scale the continuous variable to  [0,1] using the following formula:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{x_i-min(x_i)}{max(x_i)-min(x_i)}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />



### (e) Split the Data Set and Balance

* Split the dataset into 3 parts, 70% for training set, 15% for validation set and 15% fir test set
* The samples labeled as default only account for 10% of the whole dataset. Thus, the data is obviously imbalanced. Here I address this problem by using SMOTE algorithm to generate a balanced dataset.





## 4. Topic Model - LDA

#### (a) Feature Engineering

<img src="https://github.com/VitoDH/default-detection/raw/master/img/LDA.png" style="zoom:110%" />

LDA refers to **Latent Dirichlet Allocation**. In the LDA context, the process of generating a document can be viewed as follows:

* From the latent Dirichlet Distribution alpha, we obtain the the topic distribution theta of document d

* From theta, we can generate the topic z for the word in position n 
* From another  latent Dirichlet Distribution eta, generate the word distribution beta for the topic z 

* Generate the word  w_dn  from beta 



Supposed we have defined the number of topics as  **<img src="https://latex.codecogs.com/svg.latex?\Large&space;n" title="" />**, then by LDA we could obtain a topic vector for each **loan statement** denoting the the probability of the statement being assigned to each topic:



 <img src="https://latex.codecogs.com/svg.latex?\Large&space;P_m=\left\{P_{m,1},\cdots,P_{m,n}\right\}" title="" />  . 



This vector can been seen as features of topics and can be combined with the features in part 2.



The pipeline for obtaining this vector is as follows:

* Split the sentence into words (by Ansj for Chinese version), remove stopwords and meaningless noise
* Gibbs Sampling until convergence occurs
* Obtain the topic vector



#### (b) Choosing the number of topics

Here I use perplexity as the metric for selecting the number of topics. This concept is put forward by the author of LDA, Blei. Perplexity represents the ability of generalization of the model. The smaller the perplexity, the better performance in generalization.



<img src="https://github.com/VitoDH/default-detection/raw/master/img/train_perp.png" style="zoom:40%" />

From the plot above,  pick the number of topic n to be 40. The features in the topic vector are denoted as Topic1, Topic2, ..., Topic 40.





## 5. Random Forest

Random Forest is a robust classifier which is easy to interpret  and implement.



### (a) Feature Selection

In this part, I mainly select the top 10 features based on the metrics of Mean Decrease Accuracy and Mean Decrease Gini.

Here I provide the importance of features of two model:

* Model without LDA

<img src="https://github.com/VitoDH/default-detection/raw/master/img/var_imp_big.png" style="zoom:40%" />



* Model with LDA

<img src="https://github.com/VitoDH/default-detection/raw/master/img/var_imp_topic_big.png" style="zoom:40%" />



I notice that the topic features play a important role in the model. (eg, Topic 34 and Topic 1)



The 10 features  selected are as follows:

**Best Features Set**

|    Attributes    |                    Definitions                    |
| :--------------: | :-----------------------------------------------: |
|    education     | 1: high school 2: junior college 3: undergrad ... |
|  property loan   |               property of the loan                |
|  scale company   |        (# of staffs) 1: <10 2: 10-100 ...         |
| statement length |         the length of the loan statement          |
|      target      |              the amount of the loan               |
|       term       |          the holding period of the loan           |
|     time job     |            1: <1 year 2: 1-3 years ...            |
|      Topic1      |                                                   |
|     Topic34      |                                                   |
|     Topic39      |                                                   |



It's interesting to take a look at what words are included in the topics that I selected:

|     Topic 1      |    Topic 34     |     Topic 39     |
| :--------------: | :-------------: | :--------------: |
| 还款(Repayment)  |   希望(Hope)    | 投资(Investment) |
|   信用(Credit)   |    想(Want)     |  生意(Business)  |
|   工资(Salary)   | 还款(Repayment) |  资金(Capital)   |
|  逾期(Overdue)   |  谢谢(Thanks)   |   开(Start up)   |
|   房屋(House)    |   申请(Apply)   |  周转(Turnover)  |
|  能力(Ability)   |  资金(Capital)  | 有限公司(Co,Ltd) |
| 短期(Short-term) |   房子(House)   |    一家(One)     |
|  周转(Turnover)  |  支持(Support)  |   销售(Sales)    |
| 打卡(Attendence) |    钱(Money)    |   朋友(Friend)   |
|   来源(Source)   | 平台(Platform)  |    银行(Bank)    |

Note: The loan statements are all written in Chinese.

Basically, Topic 1 is relevant to the loan, Topic 34 is relevant to the attitude of borrower and Topic 39 relates to investment.



### (b) Tuning the parameters

* Select the number of trees in random forest: **ntree**

<img src="https://github.com/VitoDH/default-detection/raw/master/img/ntree_selection_topic.png" style="zoom:40%" />

When ntree is larger than 100, the error has already been steady. Thus, I choose ntree=100.



* Select the number of candidate features at each split: **mtry**

<img src="https://github.com/VitoDH/default-detection/raw/master/img/mtry_selection_topic.png" style="zoom:40%" />

The OOB error is minimized when mtry=3. Hence, I pick mtry to be 3.



* Select the number of leaves: **maxnodes**

<img src="https://github.com/VitoDH/default-detection/raw/master/img/maxnode_accuracy_topic.png" style="zoom:40%" />



<img src="https://github.com/VitoDH/default-detection/raw/master/img/maxnode_f1score_topic.png" style="zoom:40%" />



<img src="https://github.com/VitoDH/default-detection/raw/master/img/maxnode_auc_topic.png" style="zoom:40%" />



Based on the metrics accuracy, F1 score and AUC, I pick maxnode to be 400. Here, I take the complexity of the model into consideration. When maxnode is larger than 400, the improvement of the performance is not significant.



## 6. Evaluation

### (a) Test Set Performance

The ROC curve on the test set is given as:

<img src="https://github.com/VitoDH/default-detection/raw/master/img/roc_test_topic.png" style="zoom:40%" />

 

The comparison of performance on validation set and test set:

|    Metrics     | Accuracy | Precision | Recall | F1 Score |  AUC   |
| :------------: | :------: | :-------: | :----: | :------: | :----: |
| Validation Set |  0.9172  |  0.5853   | 0.9955 |  0.7372  | 0.9595 |
|    Test Set    |  0.9297  |  0.6254   | 0.9911 |  0.7668  | 0.9564 |



Noting that the test set has equivalent performance to the validation set, we can conclude that the model successfully generalizes to out-of-sample data.



### (b) Test Set Comparison Between Two Models

To further demonstrate the power of LDA, we can compare that performance of the two models on test set:

|       Metrics        | Accuracy | Precision | Recall | F1 Score |  AUC   |
| :------------------: | :------: | :-------: | :----: | :------: | :----: |
|     Without LDA      |  0.9157  |  0.5850   | 0.9550 |  0.7250  | 0.9330 |
|       With LDA       |  0.9297  |  0.6254   | 0.9911 |  0.7668  | 0.9564 |
| Percentage Increased |  1.54%   |   6.9%    | 3.77%  |  5.77%   | 2.51%  |

Each metric has been improved when we incorporate the topic vectors into the features. Thus, LDA model has a significant impact on the improvement of the feature set.







## 7. Conclusion

From the results above, we are able to conclude that:

* Model successfully captures the information in the dataset. Even without LDA, random forest has achieved satisfying accuracy on the test set.
* LDA demonstrates its power in the natural language processing. It provides insight for mining the information in the loan statement. And hence the performance of the model increases.
* The features relevant to job and the loan statement play an important role in predicting the probability of default in P2P lending.

