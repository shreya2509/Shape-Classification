# Shape-Classification

**The project aims at identifying the shape of the targets segmented from aerial images.**

* The aerial images are captured using a smartphone mounted on an Unmanned Aerial Vehicle and the targets are segmented from these images using a saliency based approach.

* Following is one of the aerial images captured during test flights. There are two targets in the image. The first one is a white semicircle with the letter X on it. The second one is a purple trapezium with the letter T on it.
![Aerial Image](https://github.com/shreya2509/Shape-Classification/blob/master/aerial.jpg)

* Following is the image after segmentation using saliency approach and grabcut                 
![Extracted Target](https://github.com/shreya2509/Shape-Classification/blob/master/semicircle.jpg)

* Main challenge in identifying the shape is that the image is pixelated, edges extracted are quite rough and the corners are not sharp and prominenet. Hence, simple naive approaches like corner detection, edge detection, contour matching to find out the shape don't give reliable results. Hence after trying these basic image processing techniques we resort to training a Convolutional Neural Network for the task.



