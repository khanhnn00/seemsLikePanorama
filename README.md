# seemsLikePanorama
Input - Browse images from which panoramic image is made from
Output - default saved at ./results

Descriptor - Since surf and sift is patented, we choose orb as our default descriptor
Match confidence - Threshold to match features
GPU acceleration - use GPU to run (NVDIA only)
Warp surface - shape of the panoramic image's surface
Blending - default is multiband, strength - 0 -> 100 (%)

After choosing, press run
-------------------------------------------------------------------------------------------
To run it simply go: python gui.py

--------------------------------------------------------------------------------------------
References:
[1]: http://matthewalunbrown.com/papers/ijcv2007.
[2]: https://github.com/opencv/opencv/blob/master/samples/python/stitching_detailed.py
