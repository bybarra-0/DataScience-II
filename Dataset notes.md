# Key Domain Insights from Peer-Reviewed Article of the Dataset

1. Multi-View Lesions and Leakage:  authors explicitly state that images do not map 1-to-1 to lesions. Images were taken at varying magnifications, angles, follow-up intervals, or with different cameras. The project decision to group on `lesion_id` in `data.py` directly honors this structure.
2. Diagnostic Class Groupings

- `bkl` (Benign Keratosis): Merges seborrheic keratosis, solar lentigo, and lichen-planus like keratosis (LPLK). The paper notes LPLKs are difficult because they frequently mimic melanoma clinically and dermascopically. This is a primary error mode to watch in confusion matrices.
- `akiec`: Combines actinic keratosis (solar keratosis) and Bowen disease (intraepithelial carcinoma), which are superficial squamous cell carcinoma variants.
- `nv`: Covers all variants of melanocytic nevi, confirmed benign by either pathology, 1.5-year digital follow-up without change, or two-expert consensus.

3. Natural Clinical Noise: authors intentionally did not remove terminal hairs, air/gel bubbles, or peripheral sun damage (solar lentigines and dilated vessels) so the dataset reflects actual clinical workflows. Preprocessing pipelines should account for these artifacts.

4. Baseline Deep Architectures: authors used an ImageNet-pretrained InceptionV3 architecture to classify and filter dermatoscopic images during dataset creation. This supports Module C's proposal to use transfer learning with standard ImageNet backbones (InceptionV3, ResNet, EfficientNet).