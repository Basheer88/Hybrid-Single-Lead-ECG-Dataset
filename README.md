# Hybrid Single-Lead ECG Dataset – Code & Reproducibility

This repository contains the code used to **construct our hybrid single-lead ECG dataset** introduced in our published paper (see link below).  
You can use this code to **reproduce the dataset from raw ECG records** or to understand exactly how it was created.

---

## How we built the dataset (in brief)

Very briefly, our pipeline does the following:

1. **Collect raw ECG data** from multiple public ECG databases.
2. **Select a single lead** (MLII / Lead II–equivalent) and **resample** all signals to a unified sampling rate.
3. **Filter and normalize** the signals to remove noise and baseline wander while preserving QRS and ST/T morphology.
4. **Detect R-peaks** and **segment fixed-length heartbeats** around each R-peak to capture the full P–QRS–T complex.
5. **Unify and relabel beat types** into a smaller set of clinically meaningful classes.
6. **Handle class imbalance** (down-sampling / over-sampling) and save the final balanced dataset in convenient formats for model training.

All these steps are implemented in the scripts/notebooks in this repository so that you can **run the same pipeline end-to-end**.

---

## Reproducibility & Resources

- 📄 **Published research article:**  
  You can read the full methodological details in our paper:  
  👉 [Link to the published article]((https://link.springer.com/article/10.1007/s13721-025-00663-6))

- 📊 **Ready-to-use dataset on Kaggle:**  
  If you don’t want to rebuild from scratch, you can directly download the final dataset here:  
  👉 [Link to Kaggle dataset](xx)

---

## How to use this repo

1. Clone this repository.
2. Follow the comments in the scripts / notebooks to:
   - Download or point to the raw ECG data.
   - Run the preprocessing and segmentation steps.
   - Export the final hybrid dataset.

Feel free to open an issue or pull request if you find bugs or want to extend the pipeline.
