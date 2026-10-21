# DS605 Lab 7 - Image Representation, PCA and Face Recognition

Eigenfaces on the AT&T (ORL) face database: 400 grayscale images (40 subjects x 10, 112x92). PCA is implemented in NumPy;
scikit-learn is used only for the classifier comparison in Part D.

## Contents

- `lab7_pca_face_recognition.ipynb` - runnable notebook (Parts A-D) with saved outputs
- `figures/` - all plots (explained variance, eigenfaces, reconstructions, recognition results, model comparison)
- `results_comparison.csv` - final comparison table
- `requirements.txt` - dependencies

**To run:** download ORL so it looks like `att_faces/s1/1.pgm ... s40/10.pgm` (not included here), run `pip install -r requirements.txt`,
set `DATA_DIR` in the first code cell, and run all cells.

## Approach

1. **Representation (A):** images read with OpenCV as grayscale, flattened to 10,304-length vectors, scaled to [0, 1]. Seeded stratified
   split: 8 train / 2 test images per subject (320 / 80).
2. **PCA (B):** mean face and PCA fit on training data only, via eigendecomposition of the 320x320 matrix and via SVD (both agree).
   `k` = smallest number of components keeping 90% variance. Reconstruction and storage compared across `k`.
3. **Recognition (C):** nearest-neighbour matching in PCA space (Euclidean and cosine) vs raw pixels.
4. **ML comparison (D):** KNN, logistic regression, SVM on PCA features; components, scaling and hyper-parameters tuned with
   4-fold cross-validation on the training set only. The test set is used once at the end.

## Results

**Variance and compression**

| Variance kept | 80% | 90% | 95% | 99% |
|---|---|---|---|---|
| Components | 42 | **99** | 164 | 267 |

At k = 99 each face goes from 10,304 to 99 numbers (104x smaller). Including the stored basis, the training set shrinks 26.38 MB to 8.50 MB (3.1x).

**Reconstruction (test set)**

| k | 5 | 10 | 20 | 50 | 99 | 200 | 319 |
|---|---|---|---|---|---|---|---|
| Cumulative variance | 0.481 | 0.601 | 0.703 | 0.823 | 0.900 | 0.968 | 1.000 |
| PSNR (dB) | 19.03 | 20.02 | 21.05 | 22.38 | 23.27 | 24.04 | 24.52 |

**Recognition and ML comparison**

| Model | Features | CV acc. (train) | Test acc. |
|---|---|---|---|
| 1-NN, raw pixels | 10,304 | - | 0.9875 |
| 1-NN, PCA (Euclidean) | 99 | - | 0.9875 |
| 1-NN, PCA (cosine) | 99 | - | 1.0000 |
| KNN (tuned) | 30 | 0.9688 | 0.9875 |
| Logistic Regression (tuned) | 50 | 0.9781 | 1.0000 |
| **SVM, linear (tuned)** | 50 | **0.9844** | 0.9875 |

![Explained variance](figures/explained_variance.png)
![Eigenfaces](figures/eigenfaces.png)
![Reconstruction examples](figures/reconstruction_examples.png)

## Observations

- 99 of 10,304 dimensions keep 90% of the variance, and PCA matching is as accurate as raw pixels (0.9875) with about 1% of the features and about 7x faster.
- The first eigenfaces are broad patterns (overall brightness, hair/forehead, lighting); later ones capture detail around the eyes, nose and mouth.
- Reconstruction improves quickly for small `k`, then slowly. Accuracy plateaus by about k = 30-50, so extra components add cost but not accuracy.
- All models score 0.96-1.0 and are effectively tied. The SVM is reported as best because it has the highest cross-validation score; logistic
  regression's 1.0 on test is one image (1.25 points) on an 80-image set.
- No sign of overfitting: CV accuracy (0.97-0.98) and test accuracy (0.99-1.0) are close.
- The only error is test face 63 (true subject 32, predicted 2), made by every method. The query is smiling with the head turned, and the match
  wears glasses; distances to the match (12.75) and to the true subject (13.00) are almost equal.
- Limitation: with 2 test images per subject, differences between models are within noise.

