import cv2 as cv
import numpy as np
import pysift
from PIL import Image
from collections import OrderedDict

# Reminder tuning variables
FEATURES_FIND_CHOICES = OrderedDict()
FEATURES_FIND_CHOICES['ORB'] = cv.ORB.create
FEATURES_FIND_CHOICES['BRISK'] = cv.BRISK_create
FEATURES_FIND_CHOICES['AKAZE'] = cv.AKAZE_create

WARP_CHOICES = (
    'spherical',
    'plane',
    'cyclindrical',
)

WAVE_CORRECT_CHOICES = ('horiz', 'vert', 'no',)

BLEND_CHOICES = ('multiband', 'feather', 'no',)

class Panorama:
    def __init__(self, output, img_names, desc, try_cuda, match_conf, wave_correct, warp, blend, blend_strength):
        self.log = ''
        self.img_names = img_names
        self.output = output
        self.work_megapix = 0.6
        self.seam_megapix = 0.1
        self.compose_megapix = -1
        self.work_scale = 0
        self.finder = FEATURES_FIND_CHOICES[desc]()
        self.seam_work_aspect = 1
        self.full_img_sizes = []
        self.features = []
        self.images = []
        self.is_work_scale_set = False
        self.is_seam_scale_set = False
        # self.matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
        self.matcher = cv.detail_BestOf2NearestMatcher(try_cuda, match_conf)

        self.is_compose_scale_set = False       
        self.p = None
        self.num_images = 0
        self.cameras = None
        self.warped_image_scale = 0
        self.corners = []
        self.masks_warped = []
        self.images_warped = []
        self.sizes = []
        self.masks = []
        self.K = None

        self.wave_correct = wave_correct
        self.warp_type = warp
        self.blend_type = blend
        self.blend_strength = blend_strength

    def clear_log(self):
        self.log = ''

    def read_images(self):
        # Check every filename
        for name in self.img_names:
            full_img = cv.imread(name)
            # If not image
            if full_img is None:
                self.log = self.log + 'Cannot read ' + name + '\n'
            else:
                # Add image shape info
                self.full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
                # No resize
                if self.work_megapix < 0:
                    img = full_img
                    self.work_scale = 1
                    self.is_work_scale_set = True
                else:
                    # Find scale on first run
                    if self.is_work_scale_set is False:
                        self.work_scale = min(1.0, np.sqrt(self.work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                        self.is_work_scale_set = True
                    # Resize
                    img = cv.resize(src=full_img, dsize=None, fx=self.work_scale, fy=self.work_scale, interpolation=cv.INTER_LINEAR_EXACT)
                # Find seam scale on first run
                if self.is_seam_scale_set is False:
                    seam_scale = min(1.0, np.sqrt(self.seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    self.seam_work_aspect = seam_scale / self.work_scale
                    self.is_seam_scale_set = True
                # Find keypoints and describe
                img_feat = cv.detail.computeImageFeatures2(self.finder, img)
                self.features.append(img_feat)
                img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
                print(img.shape)
                self.images.append(img)

    def match(self):
        # Find 2 best matches for each feature
        self.p = self.matcher.apply2(self.features)
        self.matcher.collectGarbage()
        
        # Find largest subset of images that matches, removing duplicates and outlier
        indices = cv.detail.leaveBiggestComponent(self.features, self.p, 0.3)
        img_subset = []
        img_names_subset = []
        full_img_sizes_subset = []
        # Replacing full set with subset
        for i in range(len(indices)):
            img_names_subset.append(self.img_names[indices[i, 0]])
            img_subset.append(self.images[indices[i, 0]])
            full_img_sizes_subset.append(self.full_img_sizes[indices[i, 0]])
        self.images = img_subset
        self.img_names = img_names_subset
        self.full_img_sizes = full_img_sizes_subset
        self.num_images = len(self.img_names)
        if self.num_images < 2:
            self.log = self.log + 'Not enough images\n'
            exit()
        
        # Estimate homography
        estimator = cv.detail_HomographyBasedEstimator()
        status, self.cameras = estimator.apply(self.features, self.p, None)
        if not status:
            self.log = self.log + 'Homography estimation failed\n'
            exit()
        for cam in self.cameras:
            cam.R = cam.R.astype(np.float32)
            print(cam.R.shape)
        
        # Bundle adjustment, changing camera parameters from the most matched image pair downward
        adjuster = cv.detail_BundleAdjusterRay()
        adjuster.setConfThresh(1)
        adjuster.setRefinementMask(np.zeros((3, 3), np.uint8))
        status, self.cameras = adjuster.apply(self.features, self.p, self.cameras)
        if not status:
            self.log = self.log + 'Camera adjusting failed\n'
            exit()

    def stitch(self):
        focals = []
        for cam in self.cameras:
            focals.append(cam.focal)
        sorted(focals)

        if len(focals) % 2 == 1:
            self.warped_image_scale = focals[len(focals) // 2]
        else:
            self.warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        
        # Straightening / Wave correction
        if self.wave_correct != 'No':
            rmats = []
            for cam in self.cameras:
                rmats.append(np.copy(cam.R))
            if self.wave_correct == 'Horizontal':
                rmats = cv.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_HORIZ)
            else:
                rmats = cv.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_VERT)
            for idx, cam in enumerate(self.cameras):
                cam.R = rmats[idx]
        
        # Create mask for every image
        for i in range(0, self.num_images):
            um = cv.UMat(255 * np.ones((self.images[i].shape[0], self.images[i].shape[1]), np.uint8))
            self.masks.append(um)

        # Warping at low resolution to estimate exposure and find seam mark - quality/time tradeoff
        warper = cv.PyRotationWarper(self.warp_type, self.warped_image_scale * self.seam_work_aspect)
        for idx in range(0, self.num_images):
            self.K = self.cameras[idx].K().astype(np.float32)
            self.K[0, 0] *= self.seam_work_aspect
            self.K[0, 2] *= self.seam_work_aspect
            self.K[1, 1] *= self.seam_work_aspect
            self.K[1, 2] *= self.seam_work_aspect
            corner, image_wp = warper.warp(self.images[idx], self.K, self.cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            self.corners.append(corner)
            self.sizes.append((image_wp.shape[1], image_wp.shape[0]))
            self.images_warped.append(image_wp)
            self.p, mask_wp = warper.warp(self.masks[idx], self.K, self.cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            self.masks_warped.append(mask_wp.get())
        
    def refine(self):
        # Prepare Exposure/Gain/Intensity Compensation
        compensator = cv.detail.ExposureCompensator_createDefault(cv.detail.ExposureCompensator_GAIN_BLOCKS)
        compensator.feed(corners=self.corners, images=self.images_warped, masks=self.masks_warped)

        # Convert
        images_warped_f = []
        for img in self.images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)
        # Find optimal seam line to blend
        seam_finder = cv.detail_GraphCutSeamFinder('COST_COLOR')
        seam_finder.find(images_warped_f, self.corners, self.masks_warped)

        compose_scale = 1
        self.corners = []
        self.sizes = []
        blender = None

        for idx, name in enumerate(self.img_names):
            # Read image again \\ wtf
            full_img = cv.imread(name)
            # Find scale at first image
            if self.is_compose_scale_set == False:
                if self.compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(self.compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                self.is_compose_scale_set = True
                compose_work_aspect = compose_scale / self.work_scale
                self.warped_image_scale *= compose_work_aspect
                # Prepare war   
                self.warper = cv.PyRotationWarper(self.warp_type, self.warped_image_scale)
                for i in range(0, len(self.img_names)):
                    self.cameras[i].focal *= compose_work_aspect
                    self.cameras[i].ppx *= compose_work_aspect
                    self.cameras[i].ppy *= compose_work_aspect
                    sz = (self.full_img_sizes[i][0] * compose_scale, self.full_img_sizes[i][1] * compose_scale)
                    self.K = self.cameras[i].K().astype(np.float32)
                    roi = self.warper.warpRoi(sz, self.K, self.cameras[i].R)
                    self.corners.append(roi[0:2])
                    self.sizes.append(roi[2:4])

            if abs(compose_scale - 1) > 1e-1:
                img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale, interpolation=cv.INTER_LINEAR_EXACT)
            else:
                img = full_img
            # Warp
            self.K = self.cameras[idx].K().astype(np.float32)
            corner, image_warped = self.warper.warp(img, self.K, self.cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = self.warper.warp(mask, self.K, self.cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            # Compensate exposure
            compensator.apply(idx, self.corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            # Create seam mask at original resolution
            dilated_mask = cv.dilate(self.masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            
            # Blend
            if blender is None:
                dst = cv.detail.resultRoi(corners=self.corners, sizes=self.sizes)
                blend_width = np.sqrt(dst[2] * dst[3]) * self.blend_strength / 100
                if blend_width < 1 or self.blend_type == 'No':
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif self.blend_type == 'Multiband':
                    blender = cv.detail_MultiBandBlender()
                    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
                elif self.blend_type == 'Feather':
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst)
            
            blender.feed(cv.UMat(image_warped_s), mask_warped, self.corners[idx])

        # Final image
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)

        cv.imwrite(self.output, result)
        res = Image.open(self.output)
        res.show()

def main():
    path = 'inputs3'
    imgs_name = os.listdir(path)
    panorama = panorama()
