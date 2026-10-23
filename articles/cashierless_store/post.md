---
layout: page
title: Cashierless Store
description: "A store without a checkout: ceiling cameras that follow every shopper on the floor plan, read where they stop and what they take, and send a receipt when they walk out"
img: assets/img/projects/2_cachierless_store/figures/card.png
cover: /assets/img/projects/2_cachierless_store/figures/cover.png
og_image: https://xmarva.github.io/assets/img/projects/2_cachierless_store/figures/cover.png
stylesheets: [/assets/css/tfs-course.css]
importance: 3
category: work
related_publications: false

role: ML Engineer

domains:
  - Computer Vision

tech:
  - Multi-Camera Tracking
  - Object Detection
  - Camera Calibration
  - OpenCV

website: https://digital.tatarstan.ru
---

{% raw %}

Amazon tested its first checkout-free store on its own employees for more than a year before opening it to the public in Seattle in January 2018. Reuters, citing a person familiar with the test, listed what went wrong along the way: the system had trouble "correctly identifying shoppers with similar body types", and "when children were brought into the store during the trial, they caused havoc by moving items to incorrect places" ([Reuters, 2018](https://www.reuters.com/article/us-amazon-com-store-idUSKBN1FA0RL)).

Those two sentences describe the whole problem. A store without a checkout has to know, at every moment, where each person is, and what each of them took from a shelf or put back somewhere else.

From February 2020 to January 2021 I worked on such a store for the Ministry of Digital Development of the Republic of Tatarstan, on the computer vision side: detection, tracking, the cameras and their calibration. The project started in a strange month. On 9 March 2020 Amazon began selling its technology to other retailers ([Reuters, 2020](https://www.reuters.com/article/us-amazon-com-store-technology-idUSKBN20W0OD)), and two days later the WHO declared COVID-19 a pandemic.

The store's data stays private, so the figures on this page come from a simulation built for it: a 12 by 8 metre shop with twelve ceiling cameras, five shoppers walking between the shelves, and the same geometry, detection and tracking steps the real system used.

## A store seen from the ceiling

[[layout]]

Cameras in such a store hang from the ceiling and look down at an angle, so that a person is seen from above, where shoulders and heads rarely hide each other, and still from the side, where the shelves are. In the simulation there is one camera at each end of every aisle, looking along it, and two more over the door and the front corridor.

The colours show how many cameras see a person standing at each point. Almost all of the floor, 89%, is seen by at least two cameras, and the middle of the store by up to nine. The corners next to the walls are the weak spots, and in a real store they are also where people stand to read a label.

Amazon did not publish its camera count, but Reuters described the Go Grocery store, opened in February 2020, as watched by "hundreds of ceiling cameras" ([Reuters, 2020](https://www.reuters.com/article/us-amazon-com-store-idUSKCN20J0XK)).

[[views]]

Each camera runs a person detector on every frame, here five frames a second. The detector returns a box around each person it sees, and everything else on this page is built from those boxes.

In 2020 the detectors were changing fast: YOLOv4 appeared in April ([Bochkovskiy et al., 2020](https://arxiv.org/abs/2004.10934)) and the YOLOv5 code in May. For a fixed set of cameras in one room, the bigger difference came from training on frames from those cameras, with their angles and their lighting.

## From a pixel to a point on the floor

A box in a picture is not a position in the store. To follow a person across cameras, every detection has to be carried to the same place: the floor plan.

For a pinhole camera, a point $(x, y, z)$ of the store lands at the pixel $(u, v)$ given by

$$
\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K \, [\, R \mid t \,] \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix},
$$

where $K$ holds the focal length and the image centre, $R$ and $t$ are the camera's rotation and position, and $\lambda$ is the depth. One picture cannot undo this in general: a pixel is a whole ray of points.

On a plane it can. Every point of the floor has $z = 0$, the third column of $R$ drops out, and what is left is a 3 by 3 matrix, a homography:

$$
\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = \underbrace{K \, [\, r_1 \;\; r_2 \;\; t \,]}_{H_0} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}, \qquad \begin{pmatrix} x \\ y \\ 1 \end{pmatrix} \propto H_0^{-1} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}.
$$

A homography has eight degrees of freedom, so four marked points on the floor, seen in the picture and measured on the plan, fix it ([Hartley and Zisserman, 2004](https://www.robots.ox.ac.uk/~vgg/hzbook/)). In practice more points are marked and fitted by least squares, and each lens is first calibrated with a printed checkerboard to undo its distortion ([Zhang, 2000](https://doi.org/10.1109/34.888718)).

[[homography]]

The point to carry is where the person touches the floor: the middle of the bottom of the box. DukeMTMC, the largest multi-camera dataset of the time, annotated exactly this, "each person's foot contact point with the ground", with homographies from each camera to the ground plane ([Ristani et al., 2016](https://arxiv.org/abs/1609.01775)).

The same algebra works for any horizontal plane. At height $h$ the third column comes back as $h\,r_3$, giving $H_h = K\,[\,r_1 \;\; r_2 \;\; h\,r_3 + t\,]$. That matters in a store, because the feet are often hidden.

## Twelve guesses of one position

[[fusion]]

When a shelf, a cart or another person covers the legs, the bottom of the box is not the feet, and the floor point lands far behind the person. The detector does not say so. The system has to notice it: a box whose bottom stops well above where a whole person of normal height would end is cut off.

For such boxes the top is used instead, carried by $H_h$ with an assumed height of 1.72 m. A shorter or taller shopper is then placed a little too far or too near along the camera's line of sight, an error of tens of centimetres instead of metres.

[[errors]]

Of 5,375 detections in the simulation, 57% showed the whole person. Their feet put the person a median 26 cm from the truth; the head, with the assumed height, 31 cm; and the bottom of a cut box, 117 cm, which is why it must never be used.

The errors of different cameras point along different lines of sight, so they partly cancel. The median of all cameras' points for one person lands a median 15 cm from the truth, and within 26 cm nine times out of ten.

## Following a person through the store

A position in every frame is not yet a person. The tracker has to say that the point here now is the same shopper as the point half a metre away a fifth of a second ago, and keep saying it for the whole visit.

Each track is a Kalman filter with a constant-velocity model ([Kalman, 1960](https://doi.org/10.1115/1.3662552)): it predicts where the person will be in the next frame, with an uncertainty that grows while they are unseen. New points are assigned to predicted tracks by the Hungarian method, which finds the matching with the smallest total distance ([Kuhn, 1955](https://doi.org/10.1002/nav.3800020109)).

This is the scheme of SORT, which matched boxes in the image ([Bewley et al., 2016](https://arxiv.org/abs/1602.00763)). DeepSORT added an appearance descriptor from a re-identification network and reported that it cut identity switches by 45% ([Wojke et al., 2017](https://arxiv.org/abs/1703.07402)). Our tracker worked on the floor plan instead of the image: every camera's points are matched to the same tracks, and each track keeps a running appearance vector, as in DeepSORT.

The floor plan gives the tracker one more piece of knowledge an image tracker does not have: people only appear and disappear at the door. A track that loses its person in the middle of the store waits for them, for up to twelve seconds in the simulation, instead of ending; a new track can only start near the entrance.

[[tracks]]

In the simulation the tracker follows all five shoppers for 99.7% of the time they are in the store, with a mean error of 20 cm. It makes four identity switches, in two short episodes near the door: a second track briefly starts for a shopper who is already being followed, takes over for a second or two, and is dropped.

[[handoff]]

The strip shows why the problem is called multi-camera. A shopper is seen by about six cameras on average, never by none, but the set of cameras changes every few seconds as they turn into an aisle. Nothing in the tracker depends on which cameras those are, which is the point of tracking on the floor.

Benchmarks of the time were mostly about crowds in the street. MOT20, released in March 2020, has scenes where "the density can reach values of 246 pedestrians per frame" ([Dendorfer et al., 2020](https://arxiv.org/abs/2003.09003)). A store is the opposite: few people, many occluders, and a demand that the identity never be lost, because a lost identity is a receipt sent to the wrong person.

## Who took what

Knowing where every shopper is gives the second half of the problem its context. A product leaves a shelf when a person stands in front of it and a hand goes in.

[[visits]]

A track that slows below 0.3 m/s for a second and a half is a stop. In the simulation all nine planned stops are found, each within a second of when it really began, and the stop's position names the nearest shelf.

The floor alone is not enough to say which shelf, though. Shopper 4's first stop is in the corner between the back wall and the fridges, and the nearest shelf by distance is the wrong one. In an aisle the person may face either gondola. That is decided by the cameras that see the shelf and the hand, and by what disappears from it.

Actions at a shelf were studied on overhead video: the MERL Shopping dataset labelled "Reach to Shelf", "Retract from Shelf", "Hand in Shelf", "Inspect Product" and "Inspect Shelf" ([Singh et al., 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Singh_A_Multi-Stream_Bi-Directional_CVPR_2016_paper.html)).

Recognizing the product itself is a fine-grained problem, with thousands of similar packages, rotated, half hidden, packed next to each other. SKU-110K collected 11,762 shelf images with an average of 147 products each, because "man-made scenes can be densely packed" ([Goldman et al., 2019](https://arxiv.org/abs/1904.00853)).

Amazon put weight sensors into its shelves for the same reason ([Reuters, 2018](https://www.reuters.com/article/us-amazon-com-store-idUSKBN1FA0RL)). A camera sees which shopper reached in; the shelf knows how much it lost.

## The system

[[pipeline]]

The top row of the diagram is what this page has shown: frames from every camera, people detected in each, put on the floor and tracked. The bottom row turns tracks into purchases: stops at shelves, takes and put-backs, the product, and a virtual cart per track that becomes a receipt at the door.

Detection is the heavy part, every camera at several frames a second on GPUs; everything on the floor plan, from fusion to tracking, is light enough for one process. The fragile part is the geometry. A homography is only as good as the points it was fitted on, and a camera nudged by a ladder moves every point it reports, so the calibration has to be checked, not done once.

These stores also raise the obvious question of privacy. None of the steps above needs a face: people are followed by position and by the look of their clothes, and the companies building such stores in 2019 were "finding ways to do without facial recognition" ([AP, 2019](https://www.foxbusiness.com/features/correction-grab-and-go-shopping-story)).

## References

Tracking:

- Kalman. [A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552). J. Basic Engineering, 1960.
- Kuhn. [The Hungarian method for the assignment problem](https://doi.org/10.1002/nav.3800020109). Naval Research Logistics Quarterly, 1955.
- Bewley et al. [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763). ICIP 2016.
- Wojke, Bewley and Paulus. [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402). ICIP 2017.
- Ristani et al. [Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking](https://arxiv.org/abs/1609.01775). ECCV Workshops 2016.
- Dendorfer et al. [MOT20: A benchmark for multi object tracking in crowded scenes](https://arxiv.org/abs/2003.09003). 2020.

Geometry and detection:

- Zhang. [A flexible new technique for camera calibration](https://doi.org/10.1109/34.888718). TPAMI, 2000.
- Hartley and Zisserman. [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/). 2004.
- Bochkovskiy, Wang and Liao. [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934). 2020.

Shelves and products:

- Goldman et al. [Precise Detection in Densely Packed Scenes](https://arxiv.org/abs/1904.00853). CVPR 2019.
- Singh et al. [A Multi-Stream Bi-Directional Recurrent Neural Network for Fine-Grained Action Detection](https://openaccess.thecvf.com/content_cvpr_2016/html/Singh_A_Multi-Stream_Bi-Directional_CVPR_2016_paper.html). CVPR 2016.

{% endraw %}
