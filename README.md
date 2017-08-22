# NewsAggregator

## Introduction
Text classification is a widely covered problem in Natural Language Processing. Concretely, applications of text classification such as sentiment analysis or news aggregation are widely used in commercial products.

Specifically, in this study we will focus on multiclass news classification. This is a harder problem than sentiment analysis, which is normally a binary class classification problem. Several algorithms and setups will be tested in order to see which one is the one that works better in performing this task.

All those settings will be tested on two different datasets with different number of samples and classes to obtain a more general result. On each dataset, 10-fold cross validation will be used to obtain multiple measures of performance. We will use the F1 score as a measure of performance, a common measure in measuring accuracy of NLP algorithms. From those measures obtained, a 95% confident interval will be computed using t-test in the difference of results so to make sure con precision which is the best algorithm.

## Methods
Different models will be used to test which one performs better in this kind of problem. 

Those models are build regarding both the algorithm used and the number of features that feeds each algorithm. Naive Bayes will be selected as a baseline and then we will test how a much more complicated algorithm such as MaxEnt performs against it.

Regarding the features, three different settings will be implemented:
1. Firstly, we will test all the algorithms using just a bag of words representation in which sample is mapped to a set of features in which each feature represents one word in our vocabulary and its value is the number of times that feature appears in the document. Tfidf transformation is performed to the bag of words so that the prediction is not dominated by high counts of words that common in most documents (e.g., the). With tfidif, counts of the important words (those that are less frequent) are weighted more. This setup is the simplest but it has been proved to be very effective in this kind of problems.
2. The second setting will be built by adding to that word representation some indicator features. We will define several indicator functions that for each documents focus on one aspect of it and returns either a boolean value (whether that aspect appears or not) or an integer (the number of times it occurs).
3. Lastly, in the third setup the top 100 bigrams (those 100 bigrams with higher number of counts) will be added as features to the last setting. This way we can see whether this extra information improves the performance of the model.

## Datasets
wo main datasets has been selected to use in this study:
* BBC dataset. This dataset comprises more than 2000 samples of articles from the BBC. Regarding the number of classes, the articles belong to five different classes: (business, entertainment, politics, sport, tech). This dataset has been taken from [1].

* 20newsgroup. This dataset, taken from the UCI repository, is build from around 20000 articles of 20 different classes. It is conveniently built into scikit-learn so different subsets can be chosen from it (e.g, choosing only specific classes from the whole set of categories). In this case, the whole dataset will be used. More information on this dataset can be found in [2].

[1]: D.GreeneandP.Cunningham,“Practical solutions to the problem of diagonal dominance in kernel document clustering,” in *Proceedings of the 23rd international conference on Machine learning*, pp. 377– 384, ACM, 2006.

[2]: T. Mitchell, “UCI machine learning repository,” 2013.

