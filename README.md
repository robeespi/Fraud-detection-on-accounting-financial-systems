# 1. Fraud-detection-on-accounting-financial-systems

# 1.1 Problem:

<p align=justify>Existing unsupervised fraud detection methods successfully learn representations of normal-anomaly accounting statements offering acceptable interpretability of these models. 

<a id="1">[1]</a> 
https://arxiv.org/pdf/1908.00734.pdf  Schreyer, Marco.  Sattarov, Timur   Schulze, Christian  Reimer, Bernd Borth,Damian(2019). 
Detection of Accounting Anomalies in the Latent Space using Adversarial Autoencoder Neural Networks 
KDD-ADF ’19, August 05, 2019, Anchorage, Alaska

<a id="2">[2]</a> 
https://arxiv.org/pdf/1709.05254.pdf  Schreyer, Marco.  Sattarov, Timur   Dengel, Andreas   Reimer, Bernd Borth,Damian(2019). 
Detection of Anomalies in Large-Scale Accounting Data using Deep Autoencoder Networks

<p align=justify>However, fraudster always find new ways to violate known scenarios leading unsupervised approaches to suboptimal performance. As a result, this notebook explore semi-supervised and weakly supervised approach to detect fraud.

    + Semi-supervised: A fraction of the anomalies are identified and then labelled, all clases of anomalies are identified.
    + Weakly-supervised:  A fraction of the anomalies are identified and then labelled, but not all clasess of anomalies are identified.

# 1.2 Solution:

<p align=justify>This repo contains a notebook with a solution which proposes the following workflow before to apply a Multilayer Perceptron Neural Network

![alt text](https://github.com/robeespi/Fraud-detection-on-accounting-financial-systems/blob/master/Data_preparation_workflow.jpeg)

# 2. Highlights workflow

<p align=justify>PCA helps to visualize two types of fraud class on high dimensional data. According to the domain knowledge, by having two types of anomalies make sense

* Global Anomalies: 
<p align=justify>Financial statementes that exhibit unusual or rare individual attribute values. These anomalies usually relate to highly skewed attributes e.g. seldom posting users, rarely used ledgers, or unusual posting times.

* Local Anomalies: 
<p align=justify>Financial statementes that exhibit an unusual or rare combination of attribute values while the individual attribute values occur quite frequently e.g. unusual accounting records.

![alt text](https://github.com/robeespi/Fraud-detection-on-accounting-financial-systems/blob/master/PCA.jpeg)

Also, t-sne was developed on tensorboard tow visualize fraud objects. The following link shows this https://youtu.be/dR1_WrjaFFE

# 3. How to use this notebook

<p align=justify>Please use it on Goggle Colab by enabling 25gb RAM. You can use this trick to increase the RAM of your service

https://towardsdatascience.com/upgrade-your-memory-on-google-colab-for-free-1b8b18e8791d

<p align=justify>You can explore the aforementioned workflow by using a table of contents, as the following image shows.

![alt text](https://github.com/robeespi/Fraud-detection-on-accounting-financial-systems/blob/master/table_contents.jpeg)

# 4. Results

<p align=justify>Even though semi-supervised approach outperforms unsupervised approach by labelling less than 1% out of availables fraud objects, weakly-supervised approach did not perform better than the unsupervised method. Additionally, weakly supervised method presented high variance due to small number of fraud objects available.Therefore, there are room for improvements in order to lift the ability of the model to predict unknown frauds.

Model | #Supervision settings | #F1 | 
--- | --- | --- | 
MLP | Semi-supervised | 82.85% | 
MLP knowing local-global unknown | Weakly-supervised| 66.66% | 
MLP knowing global-local unknown| Weakly-supervised| 59.45% | 
Autoencoder | Unsupervised | 72.51% | 
