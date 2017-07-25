# **Finding Lane Lines on the Road** 



**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # "Image References"

[grayscale]: test_images_output/gray.jpg "Grayscale"

[edges]: test_images_output/edges.jpg "Edges"

[masked_edges]: test_images_output/masked_edges.jpg "Masked Edges"

[spurious_lines]: test_images_output/spurious_lines.jpg "Noisy Lines"

[extreme_lines]: test_images_output/extreme_lines.jpg "Extreme Lines"

[final_lines]: test_images_output/solidYellowLeft.jpg "Final Lines"
[final_lines2]: test_images_output/additionalExample.jpg "Another Final Lines"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps:

1. Converted the images to grayscale.

   ![alt text][grayscale]

2. Applied a Gaussian smoothing with `5x5` kernel size to the images, and then performed Canny edge detection on these blurred images.

   ![alt text][edges]

3. Masked the edge images to remove edges outside the region of interest. The region itself is a polygon with 4 vertices calculated by function `vertices_of_region_of_interest()`.

   ![alt text][masked_edges]

4. Ran Hough transform on the masked images to detect lines on the region of interest. To obtain a good final result, the following sub-steps were done:

   - The low and high Canny thresholds were automatically calculated by function `canny_otsu_thresholds()` using [Otsu's method](http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html).

   - I modified the `draw_lines()` function by merging the lines into 2 groups: (right lane) lines with positive slope, and (left lane) lines with negative slope. The lines in each group were merged into a single line.

   - Even though they have been masked, many spurious lines were still detected which could distort the final left and right lane-lines (below). This issue was addressed by only considering lines whose slope-angle is between 20-160 degrees — the reasoning behind this approach is that the lane lines (at least in the example images and videos) will never be (near) horizontal.

     ![alt text][spurious_lines]
     ![alt text][extreme_lines]

   - To smooth the lane lines transitions (e.g. below) from one to the next frame, I averaged the point-slopes of the current frame's lane-lines with previous 24 frames'.

5. Drew left and right final lane lines in red overlayed on the original image.

    ![alt text][final_lines]

    ![alt text][final_lines2]



### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be the system cannot detect important lane lines when the car enters a junction such as a line where it should stop before the traffic light, or the lane lines of left/right traffic. This could be a problem whenever the car tries to turn sharp left or right.

Another shortcoming could be the system unable to detect lane lines at the peak of a steep hill and the road ahead is downward.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to darken non white or yellow areas before performing edge detection. This can remove unnecessary edges such as shadow of trees. Moreover, the system should not only consider left and right lane lines, but also a horizontal one.

Another potential improvement could be to have another camera that faces downward, or having a camera that can be automatically tilted downward or upward based on the angle of front and read sides of the car.