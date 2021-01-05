# EURECOM ML Projects
 A collection of projects regarding distributed machine learning, computer vision, bayesian modeling and deep learning

## [Satellite images adjustment](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)
<p align="center">
<img src="images/noise_comp.png" width="500">
</p>

 * <b>Pre-processing: de-noising</b>.
   After loading an image and adding some noise, compare the results of the application of three different filters:
    * Averaging filter
    * Median filter
    * Wiener filter
 * <b>Processing: low level feature detection</b>.
   In order to highlight the edges, three different approaches have been used and compared:
    * Gradient filter
    * Laplacian filter's zero crossings
    * Canny edge detector
    
   Then, the radon transform have been displayed, commented and compared to the Hough transform.
 * <b>Post-processing: high level detection and interpretation</b>.
   After interpreting the Radon transform points, find the orientation of the image and rotate it.
## [Music recommender system](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)

 * Data Analysis
    * Data Schema
    * Descriptive Statistics
    * Data Cleaning
    * Correlation Graphs
  * Statistical Models for Recommendations
    * Introduction to Recommender Systems
    * Collaborative Filtering
    * Alternating Least Squares
    * Distributed ALS
  * Music Recommender System
    * Using PySpark MLLib
    * Evaluating Recommendation Quality
    * Personalized Recommendations
  * Experimental Explorations
    * Outliers Removal
    * Using the logarithm of the Ratings
    * Hybrid Approach: collaborative filtering & content based recommendations
      * Users clustering
      * Features extraction
      * Mixing the approaches
  * Final Considerations
  
## [Batch, mini-batch, stochastic and distributed gradient descent](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)
<p align="center">
<img src="images/sgd_img.png" width="640">
</p>

This notebook contains multiple implementations of the gradient descent algorithm. At first, the results obtained using the scipy library are observed, then the algorithm is compared with a numpy implementation of batch gradient descent.

Furthermore, stochastic gradient descent and mini-batch stocastic gradient descent are implemented and compared with the previous one. A deep analysis is performed regarding how each algorithm's results change with parameters such as the learning rate and the number of iterations.

Finally, a distributed version of mini-batch gradient descent is implemented using PySpark and its followed by an analysis of the performance of all the algorithms in terms of dataset size.
## [K-means, k-means++ and distributed k-means](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)
<img align="right" src="images/kmeans_img.png" width="300">

As for the previous notebook, this one starts by analyzing the k-means algorithm and its implementation, then it is tested on a generated 2D dataset to have better visualization and it is compared with the sklearn implementation.

An analysis of convergence of datasets of different shapes underlines the important of centroids initialization and introduces the k-means++ technique for smart centroid initialization that is implemented too. Moreover, the elbow method to find the optimal value of the number of clusters is discussed and implemented.

Finally, a distributed version of k-means is implemented with PySpark, the new algorithm is analyzed and compared with the serial implementation.
## [Flight data analysis with SparkSQL](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)
<img align="left" src="images/flights_img.png" width="300">

The last notebook introduces the DataFrame API and its advantages with respèct to RDDs, then, DataFrames are built starting from a structured file and from an RDD.

This section is followed by the analysis of flights data using SparkSQL. Data exploration is divided in three main sections: basic queries, flight volume statistics and additional queries. Data visualization is performed using the seaborn module.

The notebook ends with a bonus question regarding the analysis of other datasets and their relation with the used one.

## [Image classification algorithms](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)

## Digital image processing - [Image filtering (Matlab)](https://github.com/giacomodeodato/EURECOM-ML-Projects/blob/main/Digital%20image%20processing/Image%20processing%20with%20Matlab/Image%20filtering/Report/improc_lab2_deodato.pdf),[Stereo images (Matlab)](https://github.com/giacomodeodato/EURECOM-ML-Projects/blob/main/Digital%20image%20processing/Image%20processing%20with%20Matlab/Stereo%20image%20processing/Report/improc_lab5_deodato_patti.pdf),[Image filtering (OpenCV)](https://github.com/giacomodeodato/EURECOM-ML-Projects/blob/main/Digital%20image%20processing/Image%20processing%20with%20OpenCV/imgprocessing.cpp)
This laboratory aims at analyzing the relation and performance of different kinds of filter in different domains:
 * Linear filtering in the frequency domain and analysis of the cutting frequency
 * Linear filtering in the spatial domain using the averaging filter varying its size and the noise intensity
 * Non linear filtering with the median filter and its performance with respect to the linear one.
 
Introduction to OpenCV, histogram equalization of the image and analysis of the different filters and edge detectors previously seen.

<img align="right" src="images/disp_map.png" width="200">

Processing of stereo images, stereo matching function to create disparity map with Sum of Absolute intensity Differences. Analysis of the result with different kernel sizes and images with different baselines. Depth computation and segmentation using the histogram of the distribution of grey values.

Finally, production of an anaglyph for 3D vision using colour filter glasses given the source images.
## [Financial risk estimation](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)
 * Introduction to Monte Carlo Simulations
 * Illustrative Example
 * Common Distributions used in MCS
 * Estimating Financial Risk of a Portfolio of Stocks
   * Terminology and Context
   * Data Preprocessing
     * Market factors and stocks
     * Missing values
     * Time alignment
   * A linear relationship between factors and stocks
   * Featurization of the factors
   * Defining distributions
   * Generating samples and calculating Value at Risk (VaR)
   * Evaluating results with backtesting method
   * Improving the distributions and the features
 * Summary
   
## [House prices regression](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)

## [Convolutional neural networks with tensorflow](https://github.com/giacomodeodato/EURECOM-ML-Projects/blob/main/Convolutional%20neural%20networks/CNN.ipynb)

## [Multi layer perceptron](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)
<p align="center">
<img src="images/mlp_img.png" width="640">
</p>

In this notebook we started building a Multi Layer Percepron using the sigmoid transfer function and Mean Squared Error loss function.
Initially we wrote down the calculation to execute one forward and one backward step on some artificial values, in order to see how the weights were updated. Then, we used numpy to implement a vectorized version of the feedforward and backprpagation algorithm and we added the methods to the NeuralNetwork class.

In the second part of the notebook we loaded the MNIST dataset and we defined a trin method for the NeuralNetwork class. In order to be more flexible we defined a general mini-batch gradient descent training so that we could compare the different performances of stochastic, batch and mini-batch gradient descent by changing the batch size (1 for stochastic; len(dataset) for batch; len(minibatch) for mini-batch).
Furthermore we tested the accuracy of neural networks with different hidden layer size and we compared and explained the results.
compare performances using networks with hidden layer of different size axplain the results

Finally we switched the trasfer function of the output layer with the softmax function and we used the cross-entropy loss, we tested the new network and underlined the improvements.

## [Introduction to python packages, pySpark and the HDFS](https://github.com/giacomodeodato/Algorithmic_Machine_Learning/blob/master/01_RecommenderSystem.ipynb)
The aim of this introductory lab is to get familiar with the Jupyter notebooks, python and its modules (Pandas, Matplotlib, Numpy). Finally this notebook contains a presentation of PySpark and how to interact with the HDFS, together with two examples of distributed code: word count and an analysis of night flights.
  
