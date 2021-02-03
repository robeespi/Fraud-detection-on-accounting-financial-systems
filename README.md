# Fraud-detection-on-accounting-financial-systems

* Problem:

Existing unsupervised fraud detection methods successfully learn representations of normal-anomaly accounting statements offering acceptable interpretability of these models. 

<a id="1">[1]</a> 
https://arxiv.org/pdf/1908.00734.pdf  Schreyer, Marco.  Sattarov, Timur   Schulze, Christian  Reimer, Bernd Borth,Damian(2019). 
Detection of Accounting Anomalies in the Latent Space using Adversarial Autoencoder Neural Networks 
KDD-ADF ’19, August 05, 2019, Anchorage, Alaska

<a id="2">[2]</a> 
https://arxiv.org/pdf/1709.05254.pdf  Schreyer, Marco.  Sattarov, Timur   Dengel, Andreas   Reimer, Bernd Borth,Damian(2019). 
Detection of Anomalies in Large-Scale Accounting Data using Deep Autoencoder Networks

However, fraudster always find new ways to violate known scenarios leading unsupervised approaches to suboptimal performance. As a result, this notebook explore semi-supervised and weakly supervised approach to detect fraud.

 + Semi-supervised: A fraction of the anomalies are identified and then labelled, all clases of anomalies are identified.
 + Weakly-supervised:  A fraction of the anomalies are identified and then labelled, but not all clasess of anomalies are identified.

* Solution:

This repo contains a notebook with a solution which proposes the following workflow before to apply a Multilayer Perceptron Neural Network

![alt text](https://github.com/robeespi/Fraud-detection-on-accounting-financial-systems/blob/master/Data_preparation_workflow.jpeg)

# Highlights workflow

PCA helps to visualize two types of fraud class on high dimensional data. According to the domain knowledge, by having two types of anomalies make sense

* Global Anomalies: 
Financial statementes that exhibit unusual or rare individual attribute values. These anomalies usually relate to highly skewed attributes e.g. seldom posting users, rarely used ledgers, or unusual posting times.

* Local Anomalies: 
Financial statementes that exhibit an unusual or rare combination of attribute values while the individual attribute values occur quite frequently e.g. unusual accounting records.

![alt text](https://github.com/robeespi/Fraud-detection-on-accounting-financial-systems/blob/master/PCA.jpeg)

Also, t-sne was developed on tensorboard tow visualize fraud objects. The following link shows this https://youtu.be/dR1_WrjaFFE

# Results

Model | #Supervision settings | #F1 | 
--- | --- | --- | 
MLP | Semi-supervised | 82.85% | 
MLP knowing local-global unknown | Weakly-supervised| 66.66% | 
MLP knowing global-global local | Weakly-supervised| 59.45% | 
Autoencoder | Unsupervised | 72.51% | 
