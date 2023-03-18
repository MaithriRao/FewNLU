---
layout: home 
---

This blog post is abput the paper FewNLU: Benchmarking State-of-the-Art Methods for Few-Shot Natural Language Understanding (https://arxiv.org/pdf/2109.12742.pdf). 

**Goal of this post**

This includes examples of several state-of-the-art techniques and data processing, as well as a standard training process and an evaluation framework for few-shot NLU.

# What is few shot learning?

Pre-trained models have been demonstrated to generalize well to new tasks in a given domain or modality. A machine learning framework called Few-Shot learning(FSL) enables a pre-trained model to generalize over new types of data(data that pre-trained model has not seen during training) using only fewer labeled samples. This comes under the category of meta learing.

# Why Few-Shot learning?

The process of collecting, annotating and validating the large amount of data is very expensive. And also there are times businesses do not have access to huge data and must depend on a small number of samples to produce the results. 
Few-Shot learning resolves the above mentioned problems in the following ways.
1. Concern of huge numbers of data is eliminated by generalizing only a few labeled samples.
2. Using the pretrained models, which is already trained on an extensive dataset, saves a lot of computational power and resources.
3. With FSL, models can also learn about uncommon categories of the data with little prior knowledge.
4. As long as the data in support and query sets are coherent, the model can be applied to  expand to other data domains even if it was pre-trained using a different distribution of data.

Some of the applications of few-shot learning are in the task of computer vision such as image classification, object and character recognition etc and also in the field of natural language processing in the task of translation, sentence completion, word similarity etc and also in the field of Robotics and audio processing and also Healthcare.
Prompting, popularized by GPT-3, has surfaced as a viable alternative input format for NLP models. Prompts typically include a pattern that asks the model to make a certain prediction and a verbalizer that converts the prediction to a class label. Many methods, including PET, iPET, and AdaPET, leverage prompts for few-shot learning.

# Why Benchmarking?

Benchmarking and evaluation are the backbones of scientific advancement in machine learning and natural language processing. It is impossible to make genuine progress or avoid overfitting to established datasets and metrics without precise and reliable benchmarks. New evaluation procedures have been developed in order to compare models reliably in a few-shot setting.

# Outline of this post

1. The problem with the existing evaluation protocol method
2. Background
3. Proposed
4. Results and analysis
5. Findings and conclusion
6. References

Prior work focussed on evaluating performance on different sets of protocols. One of them was using prefixed hyper parameters. But this has caused the risk of overestimation. Another approach was to evaluate using a small development set to select hyper parameters. But the problem with this is, splitting the small development set was unknown and at the same time not knowing which  data split strategy has made a huge difference.
In order to overcome this problem the authors et.al[1] proposed an evaluation framework for few-shot NLUs. This framework for evaluation comprises repeated processes starting from selecting a hyperparameter to selecting a data split and then training and evaluating the model. It is essential to identify a critical design decision in order to establish a strong evaluation structure, and one such decision is constructing the data splits for model selection. A new data split strategy called “Multi-Splits”. In Multi-Splits strategy is proposed, where the available labeled samples are randomly split into development  and training sets multiple times and later subsequently combining the outcomes of each data splits.

Evaluation of data split strategies: In order to evaluate different data split strategies three metrics are proposed .
Test set performance of the selected hyper-parameters: A good data split strategy must select a hyperparameter inorder to achieve a good test performance. 
Correlation between development set and true test set performance: As the small development set is used for the model selection, it is important to obtain a high correlation between the performances on the small development set over distribution of hyper parameters.
Stability with respect to number of runs K: Choosing the value of K should have minimum effect on the above two metrics i.e. performance and correlation. This effect is discussed in the coming section with the graph of standard deviation on different sets of hyper parameters.

The new data split strategy called Multi-Splits(MS), here the labeled data set is randomly divided into training and development sets using a fixed split ratio r. MS is compared with the existing data split strategies such as K-fold cross validation (CV), minimum description length (MDL), and bagging (BAG), random sampling (RAND) and model-informed splitting (MI). 


