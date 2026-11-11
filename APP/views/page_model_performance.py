"""Model Performance: the saved results of the independent evaluation of the live hand-point model (read-only)."""
import streamlit as st

import ui
from core import performance

ui.page_header("Model performance",
               "Measured results of the live hand-point model. Nothing on this page is estimated or calculated here: "
               "it only shows what the evaluation saved in MODEL_EVALUATION/landmarks.",
               icon_name="home", eyebrow="Evaluation")

data = performance.load_results()
if data is None:
    st.warning("No saved evaluation found. Run MODEL_EVALUATION/landmarks/evaluate_landmarks.py first.")
    ui.footer()
    st.stop()

pct = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"
fresh = data["fresh_independent"]
live = fresh["live_model"]
session = fresh["session"]

st.subheader("Final result: fresh independent test")
st.caption(f"{fresh['frames']} frames in {fresh['recordings']} recordings, all {live['classes_present']} signs, recorded {session['recorded']} "
           f"after the model was trained ({session['model_saved']}); never used for training or choosing a model. "
           "Same person and camera as the training recordings, so independent of the data, not of the signer.")
ci = live.get("ci")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy (per frame)", pct(live["accuracy"]), help=(f"95% interval {pct(ci[0])} to {pct(ci[1])}" if ci else None))
c2.metric("Precision (macro)", f"{live['precision_macro']:.2f}")
c3.metric("Recall (macro)", f"{live['recall_macro']:.2f}")
c4.metric("F1-score (macro)", f"{live['f1_macro']:.2f}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Top-3 accuracy", pct(live["top3_accuracy"]))
c2.metric("Recordings judged right", pct(live["recording_level"]["accuracy"]), help="One answer per recording: its most frequent frame answer.")
c3.metric("Counting 1/I and 2/V as equal", pct(live["pair_aware_accuracy"]), help="These pairs have the same hand shape; the app treats them as pairs.")
app = live["app_level"]
c4.metric("Right among accepted frames", pct(app["accuracy_of_accepted"]),
          help=f"Frames the app accepts (confidence at least {app['threshold']}): {pct(app['coverage'])} of all frames. This is accuracy against the truth.")
st.info("**Confidence is not accuracy.** "
        f"When the model is wrong it is often still sure: {pct(live['wrong_share_confident_90'])} of its mistakes had at least 90% confidence.")

per_class = live["per_class"]
rows = sorted(((k, v) for k, v in per_class.items() if v["support"] > 0), key=lambda kv: kv[1]["f1"])
left, right = st.columns(2)
with left:
    st.markdown("**Weakest signs (F1-score)**")
    for k, v in rows[:6]:
        st.write(f"{k}: F1 {v['f1']:.2f}, recall {v['recall']:.2f}")
with right:
    st.markdown("**Strongest signs (F1-score)**")
    for k, v in rows[::-1][:6]:
        st.write(f"{k}: F1 {v['f1']:.2f}, recall {v['recall']:.2f}")

st.subheader("Charts")
for caption, folder, name in (("Confusion matrix (fresh independent test)", "fresh_independent", "confusion_matrix.png"),
                              ("Precision, recall and F1 per class", "fresh_independent", "precision_recall_f1_chart.png"),
                              ("Existing validation versus the fresh test", "fresh_independent", "accuracy_chart.png"),
                              ("Live model versus the MobileNetV2 models", "fresh_independent", "model_comparison.png")):
    path = performance.chart(folder, name)
    if path:
        st.image(str(path), caption=caption, use_container_width=True)

st.subheader("Comparison on the same fresh recordings")
comparison = {"hand-point model (live)": (live["n"], live["accuracy"], live["f1_macro"])}
comparison.update({m: (r["n"], r["accuracy"], r["f1_macro"]) for m, r in fresh["image_models"].items()})
st.dataframe([{"Model": m, "Frames / pictures": n, "Accuracy": pct(a), "Macro F1": f"{f:.2f}"} for m, (n, a, f) in comparison.items()],
             hide_index=True, use_container_width=True)
st.caption("Each model is tested on its own input from the same recordings: hand data for the live model, saved hand pictures for MobileNetV2.")

st.subheader("Existing data (not final)")
st.warning("These sets were used to train the model, to choose the saved version, or share near-identical samples with training. "
           "Do not quote them as the model's accuracy.")
names = {"train_all": "Training data", "dataset_test": "Original test split (leaked)", "eraakash_holdout": "Extra-data hold-out (same signers)",
         "webcam_newest": "Your newest takes (chose the model)", "islan_signer6": "ISLAN Signer_6 (identical to training; not independent)"}
sets = data["existing_data"]["sets"]
st.dataframe([{"Set": label, "Samples": sets[k]["n"], "Accuracy": pct(sets[k]["accuracy"]), "Macro F1": f"{sets[k]['f1_macro']:.2f}"}
              for k, label in names.items() if k in sets], hide_index=True, use_container_width=True)
for caption, name in (("Training versus validation accuracy (training log)", "training_validation_accuracy.png"),
                      ("Distance to training data: how much of each set is new", "leakage_distance_chart.png")):
    path = performance.chart("existing_data", name)
    if path:
        st.image(str(path), caption=caption, use_container_width=True)

with st.expander("About this evaluation"):
    st.markdown(f"- Model: `{data['model']['file']}`, {data['model']['parameters']:,} parameters, evaluated frame by frame; evaluation generated {data['generated']}.\n"
                "- The live app adds a confidence threshold, a vote over recent frames and 1/I, 2/V pair handling on top of the model.\n"
                "- For a result that is also independent of the signer, record with another person or room and re-run the evaluation.\n"
                "- Full report: `MODEL_EVALUATION/landmarks/evaluation_summary.txt`.")
ui.footer()
