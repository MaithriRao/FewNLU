---
layout: home 
---

This blog post [FewNLU: Benchmarking State-of-the-Art Methods for Few-Shot Natural Language Understanding] (https://arxiv.org/pdf/2109.12742.pdf). 

# Goal of this post 
This includes examples of several state-of-the-art techniques and data processing, as well as a standard training process and an evaluation framework for few-shot NLU.

# What is few shot learning?
Pre-trained models have been demonstrated to generalize well to new tasks in a given domain or modality. A machine learning framework called Few-Shot learning(FSL) enables a pre-trained model to generalize over new types of data(data that pre-trained model has not seen during training) using only fewer labeled samples. This comes under the category of meta learing.

##Why Few-Shot learning?
The process of collecting, annotating and validating the large amount of data is very expensive. And also there are times businesses do not have access to huge data and must depend on a small number of samples to produce the results. 
Few-Shot learning resolves the above mentioned problems in the following ways.
Concern of huge numbers of data is eliminated by generalizing only a few labeled samples.
Using the pretrained models, which is already trained on an extensive dataset, saves a lot of computational power and resources.
With FSL, models can also learn about uncommon categories of the data with little prior knowledge.
As long as the data in support and query sets are coherent, the model can be applied to  expand to other data domains even if it was pre-trained using a sciatica;;y different distribution of data.

Some of the applications of few-shot learning are in the task of computer vision such as image classification, object and character recognition etc and also in the field of natural language processing in the task of translation, sentence completion, word similarity etc and also in the field of Robotics and audio processing and also Healthcare.
Prompting, popularized by GPT-3, has surfaced as a viable alternative input format for NLP models. Prompts typically include a pattern that asks the model to make a certain prediction and a verbalizer that converts the prediction to a class label. Many methods, including PET, iPET, and AdaPET, leverage prompts for few-shot learning.



