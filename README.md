# Predicting Potential Repeated Buyers
#### **Author**: Dehai Liu

**Department of Mathematics, Sun Yat-Sen University**



## 1. Abstract

Our project mainly focus on digging out the potential online repeated-buyers with data-driven methods and state-of-the-art algorithms. It can not only provide guidance for the sellers about their operation,but also help the online shopping platform to pinpoint the target customers for advertisements and coupons.



## 2. Data Description

`T-mall` is one of the largest online shopping platform in China.  On November 11th, "the Single Day", the trading volume on T-mall reaches 120 billion CNY, indicating tremendous profit behind this shopping festival. In this project, our data includes the shopping records of the buyers for six months before the and on the Single Day. The data is divided into **3** parts as follows:



**User Log**

| Attributes  |                      Definitions                      |
| :---------: | :---------------------------------------------------: |
|   user_id   |                  Unique ID for user                   |
|   item_id   |                Unique ID for commodity                |
|   cat_id    |        Unique ID for the category of the good         |
| merchant_id |              Unique ID for the merchant               |
|  brand_id   |                Unique ID for the brand                |
| time_stamp  |         The date of action given by customer          |
| action_type | 0:click, 1: add to cart, 2: order, 3: add to favorite |



**User Info**

| Attributes |                         Definitions                          |
| :--------: | :----------------------------------------------------------: |
|  user_id   |                      Unique ID for user                      |
| age_range  | 1: < 18, 2: [18,24], 3: [25:29], 4: [30,34], 5: [35,39], 6: [40,49], 7: >=50, 0: unknown |
|   gender   |                 0: female, 1: male, 2: NULL                  |



**Training Set and Test Set**

| Attributes  |                Definitions                 |
| :---------: | :----------------------------------------: |
|   user_id   |             Unique ID for user             |
| merchant_id |           Unique ID for merchant           |
|    label    | 1: repeated buyers, 0: non-repeated buyers |





## 3. Data Preprocessing

### (a) Load the data

* Load use_log.csv and user_info.csv
* Remove the data points including NA in user_info
* Load train_format1.csv, pick a subset of 100,000 samples and denote it as trainSet



### (b) Outlier Detection

Since the existence of click farming, we need to find out the user and the merchant with unusual clicking behavior and then remove them from the trainSet. Here we offer the scatterplot of  four actions  for users and merchants, respectively.



**User**

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/scatter_plot_1.png" style="zoom:90%" />



**Merchant(Seller)**

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/scatter_plot_2.png" style="zoom:90%" />

Based on the distribution above, we define the data to be an outlier if it exceeds the threshold below:

|                 |  User  | Merchant |
| :-------------: | :----: | :------: |
|      Click      | > 4000 | > 200000 |
|   Add to Cart   |  None  |  > 250   |
|       Buy       | > 100  | > 10000  |
| Add to favorite | >  450 | > 10000  |



After removing the outlier of users and merchants, we still have **90917** samples in training set.







## 4. Feature Engineering

### a. Direct Link

Given a specific buyer and seller, it's not difficult to find out the times of the four actions that the buyers have done in the store of the merchants. Here we define a 6-dimension vector to denote the direct link between the user and the merchant:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;v_d=(gender,age,click,cart,buy,favorite)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

### b. Indirect Link

##### (1) First we define the weight for different action types:

|     Action      | Value | Times | Weight |
| :-------------: | :---: | :---: | :----: |
|      Click      |   0   |  n_1  |  0.1   |
|   Add to Cart   |   1   |  n_2  |  0.2   |
|       Buy       |   2   |  n_3  |  0.3   |
| Add to Favorite |   3   |  n_4  |  0.4   |



##### (2) Vitality and Popularity

Vitality is used to measure the extent of how much the user loves shopping. Popularity refers to how attractive the merchant's commodity is. They are both calculated by the weighted average of the four actions. And they can be specifically defined as **category vitality**, **brand vitality** for the user and **category popularity**, **brand popularity** for the merchant.



Now we illustrate the calculation by taking the **category vitality** as an example.



The score of a good<img src="https://latex.codecogs.com/svg.latex?\Large&space;j" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> for a given user <img src="https://latex.codecogs.com/svg.latex?\Large&space;i" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> is



<img src="https://latex.codecogs.com/svg.latex?\Large&space;score_{ij}=0.1*n_1+0.2*n_2+0.3*n_3+0.4*n_4" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />



The  **category vitality** of user <img src="https://latex.codecogs.com/svg.latex?\Large&space;i" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> is



<img src="https://latex.codecogs.com/svg.latex?\Large&space;S_{cat-vitality}^i=mean(score_{ij})" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />



where <img src="https://latex.codecogs.com/svg.latex?\Large&space;n_i" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> refers the item that is relevant to the user <img src="https://latex.codecogs.com/svg.latex?\Large&space;i" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />.



Similarly, we can calculate the other three indicators and combine them into a 4-dimension vector.

  











### c. Normalization

After setting up the features, we find out the each feature have different scales. Thus, it would be reasonable to scale the attributes to [0,1] using the following formula:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{x_i-min(x_i)}{max(x_i)-min(x_i)}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />



### d. PCA

Based on part a and b, we have obtained **10** features. In order to simplify the training process and remove useless information, we perform PCA on the training set. The scree plot and the variance of components are given as follows:

 <img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/scree_plot.png" style="zoom:90%" />

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/pca_var.png" style="zoom:90%" />



Noting that the cumulative proportion has reach **0.94** on the 4th principal component, we can simply pick the first four principal components as our attributes for training.





## 5. Balance 

Taking a glance at the distribution of the label, 

|        Type        | Number |
| :----------------: | :----: |
|    Total Sample    | 90917  |
| Positive Label (1) |  5363  |
| Negative Label (0) | 85554  |
|   Number of User   | 75053  |
| Number of Merchant |  1982  |



From the above table, the positive samples only covers 5.9% of the total samples, which will easily lead to a situation that all the positive label will be classified as negative.

We use four ways of sampling to address this problem and obtain a balance dataset.



|     Label      |   0   |   1   |
| :------------: | :---: | :---: |
|      Raw       | 84700 | 5300  |
| Over Sampling  | 84700 | 84700 |
| Under Sampling | 5300  | 5300  |
| Over and Under | 44959 | 45041 |
|     SMOTE      | 44959 | 45041 |





## 6. Training with XG Boost

XG Boost is an cutting-edge algorithm derived from GBDT , which can deal with missing data and avoid overfitting.

### a. Parametrization

|    Parameters     |  Value   |
| :---------------: | :------: |
|     max_depth     |    5     |
|   learning_rate   |   0.1    |
|     max_iter      |   800    |
| learning_function | logistic |



### b. Performance under different sampling

#### (1) Over Sampling

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/xg_over_sample.png" style="zoom:90%" />

|               | **Train** | **Test** |
| :-----------: | :-------: | :------: |
| **Precision** |   0.783   |  0.128   |
|  **Recall**   |   0.846   |  0.444   |
| **F1 Score**  |   0.814   |  0.200   |
| **F2 Score**  |   0.832   |  0.297   |
|    **AUC**    |   0.885   |  0.603   |



#### (2) Under Sampling

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/xg_under_sample.png" style="zoom:90%" />

|               | **Train** | **Test** |
| :-----------: | :-------: | :------: |
| **Precision** |   0.833   |   0.08   |
|  **Recall**   |   0.862   |  0.667   |
| **F1 Score**  |   0.848   |  0.144   |
| **F2 Score**  |   0.856   |  0.270   |
|    **AUC**    |   0.929   |  0.549   |



#### (c) Both

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/xg_both.png" style="zoom:90%" />

|               | **Train** | **Test** |
| :-----------: | :-------: | :------: |
| **Precision** |   0.817   |  0.119   |
|  **Recall**   |   0.846   |  0.460   |
| **F1 Score**  |   0.832   |  0.190   |
| **F2 Score**  |   0.839   |  0.293   |
|    **AUC**    |   0.909   |  0.609   |



#### (d) SMOTE

<img src="https://github.com/VitoDH/repeated_buyers/raw/master/img/xg_smote.png" style="zoom:90%" />

|               | **Train** | **Test** |
| :-----------: | :-------: | :------: |
| **Precision** |   0.727   |  0.143   |
|  **Recall**   |   0.687   |  0.460   |
| **F1 Score**  |   0.706   |  0.218   |
| **F2 Score**  |   0.695   |  0.318   |
|    **AUC**    |   0.789   |  0.641   |



## 7. Conclusion

From the results above, we are able to conclude that:

* Model successfully captures the information in the dataset, represented by high F1 score and AUC in the training set.
* Model can be used to detect whether a buyer will come back again to a specific online store as long as the data between them is given.
* For improvement in the test set, we need to focus more on the feature engineering part and the sampling part.

