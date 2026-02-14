import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent
RESULTS_PATH = ROOT / 'reports' / 'models_all' / 'results_all.json'
MODELS_DIR = ROOT / 'reports' / 'models_all' / 'models'
ROC_DIR = ROOT / 'reports' / 'models_all'


def interpret_mcc(mcc_value):
    if mcc_value >= 0.8:
        return "Excellent agreement (0.8+)"
    elif mcc_value >= 0.6:
        return "Strong agreement (0.6-0.8)"
    elif mcc_value >= 0.4:
        return "Moderate agreement (0.4-0.6)"
    elif mcc_value >= 0.2:
        return "Fair agreement (0.2-0.4)"
    elif mcc_value >= 0:
        return "Slight agreement (0-0.2)"
    else:
        return "Poor/Inverse agreement (<0)"


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return {}
    return json.loads(RESULTS_PATH.read_text())


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig


def main():
    st.title('Bank Marketing — Model Explorer')

    results = load_results()
    if not results:
        st.warning('No results found. Run `train_all_models.py` first to generate `reports/models_all/results_all.json`.')
        return

    model_names = list(results.keys())
    selected = st.sidebar.selectbox('Select model', model_names)

    metrics = results[selected]
    st.header(selected.replace('_', ' ').title())

    # Metrics
    st.subheader('Metrics')
    metrics_display = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    st.json(metrics_display)

    # MCC interpretation
    if 'mcc' in metrics:
        st.markdown(f"**MCC:** {metrics['mcc']:.4f} — {interpret_mcc(metrics['mcc'])}")

    # Confusion matrix
    if 'confusion_matrix' in metrics:
        st.subheader('Confusion Matrix')
        cm = metrics['confusion_matrix']
        fig = plot_confusion(cm)
        st.pyplot(fig)

    # ROC image if available
    roc_path = ROC_DIR / f'roc_{selected}.png'
    if roc_path.exists():
        st.subheader('ROC Curve')
        st.image(str(roc_path))

    # Load model for predictions
    model_file = MODELS_DIR / f'{selected}.joblib'
    if model_file.exists():
        st.subheader('Batch Predictions')
        uploaded = st.file_uploader('Upload CSV for batch prediction', type=['csv'])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                model = load_model(model_file)
                # Attempt prediction
                preds = model.predict(df)
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(df)[:, 1]
                    out = pd.DataFrame({'prediction': preds, 'probability': probs})
                else:
                    out = pd.DataFrame({'prediction': preds})
                st.write(out.head(50))
                st.download_button('Download predictions CSV', out.to_csv(index=False), file_name='predictions.csv')
            except Exception as e:
                st.error(f'Error running predictions: {e}')
    else:
        st.info('Trained model artifact not found. Run training scripts to produce model files.')


if __name__ == '__main__':
    main()
