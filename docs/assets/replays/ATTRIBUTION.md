# Source attribution

## MVTec AD

MVTec Software GmbH. [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Source image and mask names, additional defect donors where recorded, and
modification descriptions are listed per image in image_manifest.jsonl.
Original embedded attribution metadata is preserved.

## CAMELYON16

Radboud University Medical Center, CAMELYON dataset.
[Source and CC0 declaration](https://registry.opendata.aws/camelyon/).
Slide identifiers and available crop coordinates are recorded in the manifest.
Mosaic extracts slide regions and thumbnails and, in the contrast task, adds
synthetic gray patches. Source CC0 status is unchanged.

## SmartDoc15-CH1

Joseph Chazalon and Marçal Rusiñol, SmartDoc Challenge 1 dataset.
[Dataset record](https://zenodo.org/records/1230218),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Video/frame references appear in the manifest; frames are extracted and encoded
for benchmark use, with homography questions derived from corner annotations.
Required citation: Jean-Christophe Burie, Joseph Chazalon, Mickaël Coustaty,
Sébastien Eskenazi, Muhammad Muzzamil Luqman, Maroua Mehri, Nibal Nayef,
Jean-Marc Ogier, Sophea Prum and Marçal Rusiñol. “ICDAR2015 Competition on
Smartphone Document Capture and OCR (SmartDoc).” 13th International Conference
on Document Analysis and Recognition, 2015.
[Paper](https://doi.org/10.1109/ICDAR.2015.7333943).

## Synthetic illustrations

Mosaic authors. Programmatically generated optical illusions, CC BY 4.0.
Images may contain labels rendered with DejaVu Sans; no font software is bundled.

## CARPK

Meng-Ru Hsieh, Yen-Liang Lin and Winston H. Hsu. Drone-based object counting by
spatially regularized regional proposal networks. ICCV 2017.
[Project](https://lafi.github.io/LPN/),
[official EULA](https://docs.google.com/forms/d/e/1FAIpQLSdvwH0U_m2eTFpcPWb07y9kLmrstkymI0qBFdHXWMmE3pdxBw/viewform),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Only CARPK drone images are used, not PUCPR+ images. Mosaic derives counts and
grid questions from the source boxes and re-encodes selected images.

## BDD100K

Copyright ©2018. The Regents of the University of California (Regents).
All Rights Reserved. See BDD100K-LICENSE.txt for the complete required notice.
[Official data and annotation license](https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/license.rst).
Images are selected/re-encoded; query points and answers derive from drivable masks.
Research/nonprofit distribution conditions apply; no general commercial grant
is made by Mosaic.

## Tsinghua-Tencent 100K

Zhe Zhu, Dun Liang, Songhai Zhang, Xiaolei Huang, Baoli Li and Shimin Hu.
Traffic-Sign Detection and Classification in the Wild. CVPR 2016.
[Official source and CC-BY-NC declaration](https://cg.cs.tsinghua.edu.cn/traffic-sign/).
The source does not specify a CC license version; none is invented here.
Selected images are re-encoded; mirror-sign examples are cropped and horizontally
flipped. Sign labels supply speed-limit answers. Noncommercial use only.

## COCO references (no images bundled)

All COCO-derived images are excluded from this package. The manifest provides
original COCO filenames, COCO image IDs, original-image URLs and Flickr photo
page references. Known attribution and licenses are retained where available.
A source URL is not a grant of permission to redistribute an image or adaptation.

## LEVIR-CD

Hao Chen and Zhenwei Shi. [LEVIR-CD](https://justchenhao.github.io/LEVIR/).
Source imagery comes from Google Earth. The source specifies academic use and
prohibits commercial use, and requires compliance with Google Earth terms.
Mosaic creates perspective warps, tile arrangements and change-localization tasks.
A separate public rehosting grant has not been established by this package.

## Rico

[Rico dataset copyright policy](https://www.interactionmining.org/legal/copyright).
Screenshots may contain third-party copyrighted content; source policy places
responsibility on users and conditions access by research associates on acceptance
of its terms. Mosaic creates screenshot pairs with synthetic UI modifications.
No blanket license over screenshot content or UI annotations is granted here.

## Mirror Segmentation Dataset (MSD)

Xin Yang, Haiyang Mei, Ke Xu, Xiaopeng Wei, Baocai Yin and Rynson W.H. Lau.
Where Is My Mirror? ICCV 2019.
[Dataset project](https://mhaiyang.github.io/ICCV2019_MirrorNet/index).
Mosaic selects mirror-region tasks and constructs synthetic image differences.
An explicit dataset redistribution grant was not established; the software
repository license is not presented as an image license.

## PubLayNet

[IBM PubLayNet license](https://github.com/ibm-aur-nlp/PubLayNet/blob/master/LICENSE.md).
Annotations are CDLA-Permissive-1.0. Images retain individual PMC article terms.
Mosaic rotates document pages and constructs angle-choice tasks. Article-level
image attribution and permission have not been established for these page IDs.
