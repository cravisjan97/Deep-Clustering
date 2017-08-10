# Deep-Clustering
A Convolutional Neural Network based model for Unsupervised Learning

# Results
Here is a comparison plot of K-Means and our CNN based model on 2D data generated from two Gaussian samples
<img src="./writeup/input plot.jpg" width="265"/> <img src="./writeup/kmeans plot.jpg" width="265"/> <img src="./writeup/nn output.jpg" width="265"/> <br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Input Plot &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; K-Means Plot&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;NN Plot <br/>

We next performed Clustering on MNIST dataset(784 dimensional data) and extracted a few samples from each cluster
<img src="./writeup/image.jpg" width="2000" />
<img src="./writeup/image_2.jpg" /><br/>
Our CNN based model far exceeded the K-Means algorithm in NMI score<br/>
<img src="./writeup/table.jpg" />
