# Virtual-Classroom-Engagement-Detector

ML-based system to detect student engagement in virtual classrooms using facial emotion recognition.

# Facial Emotion Recognition

Dataset used: [Kaggle Dataset](https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset)

To run the notebook:

1. Upload your `kaggle.json` API token.
2. Run the cell to download and unzip the dataset.

## Day 1 (One-week plan) - Completed Setup

Use [notebooks/engagement_detector.ipynb](notebooks/engagement_detector.ipynb) for Day 1 tasks:

- Dataset path setup
- Label and file scan
- Class imbalance chart
- Random sample visualization per class

Before running:

1. Select a Python kernel in VS Code.
2. Update `DATASET_ROOT` in the notebook to your extracted dataset path.
3. Run all cells and keep output screenshots for your proposal/report.

## Day 2 to Day 6 (Notebook workflow)

Continue in [notebooks/engagement_detector.ipynb](notebooks/engagement_detector.ipynb):

- Day 2: Face detection + preprocessing pipeline
- Day 3: Algorithm 01 - SVM
- Day 4: Algorithm 02 - Random Forest
- Day 5: Class imbalance handling
- Day 6: Final evaluation and report exports

## Day 7 (UI Demo)

1. Run notebook cells up to Day 7 export (creates model artifacts in `artifacts/`).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start Streamlit demo:

   ```bash
   streamlit run app.py
   ```

4. Upload an image with a clear frontal face to get emotion prediction.
