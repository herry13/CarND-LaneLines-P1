# **Finding Lane Lines on the Road** 



**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # "Image References"

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps:

1. Converted the images to grayscale **TODO: an example image**
2. Applied a Gaussian smoothing with `5x5` kernel size to the images, and then performed Canny edge detection on these blurred images. **TODO: an example image**
3. Masked the edge images to remove edges outside the region of interest. The region itself is a polygon with 4 vertices: bottom-left corner, slightly-left center, slightly-right centre, and bottom-right corner. **TODO: show an example image**
4. Ran Hough transform on the masked images to detect lines on the region of interest. To obtain a good final result, the following sub-steps were done:

   - The low and high Canny thresholds were automatically calculated using [Otsu's method](http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html).
   - I modified the `draw_lines()` function by merging the lines into 2 groups: (right lane) lines with positive slope, and (left lane) lines with negative slope. The lines in each group were merged into a single line.
   - Even though they have been masked, many spurious lines were still detected which could distort the final left and right lane-lines (**TODO: example image**). This issue was addressed by only considering lines whose slope-angle is between 20-160 degrees — the reasoning behind this approach is that the lane lines will never be (near) horizontal.
   - To smooth the lane lines transitions from one to the next frame, I averaged the point-slopes of the current frame's lane-lines with previous 24 frames'.

5. Drew left and right lane lines in red overlayed on the original image.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
