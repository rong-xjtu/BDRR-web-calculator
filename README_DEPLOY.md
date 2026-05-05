# BDRR Web Calculator — ready-to-deploy package

This package is a browser-based Streamlit calculator for the BDRR model.

## Model included

The exported model files have already been placed in:

```text
saved_models/best_model_20260121_161844/
  best_model.pkl
  preprocessing_pipeline.pkl
  model_metadata.json
  feature_importance.csv
  data_statistics.json
  README.md
```

Model metadata:

- Model type: Gradient Boosting / GradientBoostingClassifier
- Test AUC: 0.8533
- Test accuracy: 0.8155
- Feature count: 10
- Training timestamp: 20260121_161844

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Deploy online with Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the contents of this folder to the repository.
3. In Streamlit Community Cloud, choose the repository.
4. Set the app entrypoint to:

```text
app.py
```

5. Deploy.

## Important compatibility note

The uploaded model pickle was saved with scikit-learn 1.3.2.  
Therefore, `requirements.txt` pins `scikit-learn==1.3.2`, `numpy==1.26.4`, and Python 3.11 via `runtime.txt`.

Do not casually upgrade scikit-learn for this deployed app unless you re-export the model under the new version.

## Batch prediction

A template file is included:

```text
sample_input.csv
```

The app accepts CSV/XLSX files and returns downloadable prediction results.

## Clinical-use note

This tool is intended for research and decision-support use only.  
It should not replace the 21-gene assay, multidisciplinary assessment, or guideline-based clinical judgment.
