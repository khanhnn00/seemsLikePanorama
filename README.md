# seemsLikePanorama
Input - Browse images from which panoramic image is made from<br/>
Output - default saved at ./results<br/>

Descriptor - Since surf and sift is patented, we choose orb as our default descriptor<br/>
Match confidence - Threshold to match features<br/>
GPU acceleration - use GPU to run (NVDIA only)<br/>
Warp surface - shape of the panoramic image's surface<br/>
Blending - default is multiband, strength - 0 -> 100 (%)<br/>

After choosing, press run

To run this repository, simply go: python gui.py<br/>

successfully folder is containing all the case that our application can perform panoramic image<br/>

References:<br/>
[1]: http://matthewalunbrown.com/papers/ijcv2007.<br/>
[2]: https://github.com/opencv/opencv/blob/master/samples/python/stitching_detailed.py
