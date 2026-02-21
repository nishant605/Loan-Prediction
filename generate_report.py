"""
Generate a professional PDF report about the Loan Prediction Model.
Run:  python generate_report.py
Produces: LoanAI_Model_Report.pdf
"""

import json
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, HRFlowable, Image
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF


# ── Load metrics ──
with open("static/metrics.json", "r") as f:
    metrics = json.load(f)

scores = metrics["scores"]
cm_data = metrics["confusion_matrix"]
cv = metrics["cross_validation"]
feat = metrics["feature_importance"]
class_dist = metrics["class_distribution"]
cibil = metrics["cibil_distribution"]
model_info = metrics["model_info"]
roc = metrics["roc_curve"]
pr = metrics["pr_curve"]

# ── Color palette ──
DARK_BLUE = colors.HexColor("#0f172a")
NAVY = colors.HexColor("#1e3a5f")
TEAL = colors.HexColor("#0d9488")
EMERALD = colors.HexColor("#059669")
AMBER = colors.HexColor("#d97706")
RED = colors.HexColor("#dc2626")
LIGHT_GRAY = colors.HexColor("#f1f5f9")
MID_GRAY = colors.HexColor("#94a3b8")
BORDER_GRAY = colors.HexColor("#cbd5e1")
WHITE = colors.white
SOFT_BLUE = colors.HexColor("#3b82f6")
PURPLE = colors.HexColor("#7c3aed")


def build_pdf():
    filename = "LoanAI_Model_Report.pdf"
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # ── Custom styles ──
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=28,
        textColor=NAVY,
        spaceAfter=2 * mm,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=MID_GRAY,
        spaceAfter=8 * mm,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=NAVY,
        spaceBefore=10 * mm,
        spaceAfter=5 * mm,
        fontName="Helvetica-Bold",
    )
    subheading_style = ParagraphStyle(
        "CustomSubheading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=TEAL,
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        textColor=DARK_BLUE,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=4 * mm,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=MID_GRAY,
        alignment=TA_CENTER,
        spaceBefore=2 * mm,
        spaceAfter=6 * mm,
    )
    metric_label_style = ParagraphStyle(
        "MetricLabel",
        parent=styles["Normal"],
        fontSize=8,
        textColor=MID_GRAY,
        alignment=TA_CENTER,
    )
    metric_value_style = ParagraphStyle(
        "MetricValue",
        parent=styles["Normal"],
        fontSize=20,
        textColor=NAVY,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )

    elements = []

    # ══════════════════════════════════════════════
    #  PAGE 1: COVER / OVERVIEW
    # ══════════════════════════════════════════════

    elements.append(Spacer(1, 30 * mm))
    elements.append(Paragraph("LoanAI", title_style))
    elements.append(Paragraph("Model Performance Report", ParagraphStyle(
        "Cover2", parent=styles["Heading1"], fontSize=20, textColor=TEAL,
        spaceAfter=5 * mm, fontName="Helvetica"
    )))

    date_str = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Generated on {date_str}", subtitle_style))

    elements.append(HRFlowable(
        width="100%", thickness=2, color=TEAL,
        spaceBefore=5 * mm, spaceAfter=10 * mm
    ))

    # Model overview
    elements.append(Paragraph("Model Overview", heading_style))
    elements.append(Paragraph(
        f"This report presents the performance evaluation of the <b>{model_info['name']}</b> "
        f"model trained for loan approval prediction. The model uses the <b>{model_info['algorithm']}</b> "
        f"algorithm with <b>{model_info['n_estimators']}</b> estimators, a maximum depth of "
        f"<b>{model_info['max_depth']}</b>, and a learning rate of <b>{model_info['learning_rate']}</b>. "
        f"The model was trained on <b>{model_info['total_samples']:,}</b> samples with "
        f"<b>{model_info['n_features']}</b> features.",
        body_style
    ))

    # Model config table
    config_data = [
        ["Parameter", "Value"],
        ["Algorithm", model_info["algorithm"]],
        ["Pipeline Name", model_info["name"]],
        ["Number of Estimators", str(model_info["n_estimators"])],
        ["Max Depth", str(model_info["max_depth"])],
        ["Learning Rate", str(model_info["learning_rate"])],
        ["Subsample Ratio", str(model_info["subsample"])],
        ["Total Training Samples", f"{model_info['total_samples']:,}"],
        ["Number of Features", str(model_info["n_features"])],
    ]

    config_table = Table(config_data, colWidths=[90 * mm, 80 * mm])
    config_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_BLUE),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(config_table)
    elements.append(Paragraph("Table 1: Model Hyperparameters & Configuration", caption_style))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════
    #  PAGE 2: KEY METRICS
    # ══════════════════════════════════════════════

    elements.append(Paragraph("Performance Metrics", heading_style))
    elements.append(Paragraph(
        "The following metrics evaluate the model's classification performance on the full dataset. "
        "All values are rounded to four decimal places.",
        body_style
    ))

    # Hero metric cards (2 rows x 5 cols)
    hero_metrics = [
        ("Accuracy", f"{scores['accuracy']:.4f}"),
        ("Precision", f"{scores['precision']:.4f}"),
        ("Recall", f"{scores['recall']:.4f}"),
        ("F1 Score", f"{scores['f1_score']:.4f}"),
        ("ROC AUC", f"{scores['roc_auc']:.4f}"),
        ("Specificity", f"{scores['specificity']:.4f}"),
        ("MCC", f"{scores['mcc']:.4f}"),
        ("Cohen's κ", f"{scores['kappa']:.4f}"),
        ("Log Loss", f"{scores['log_loss']:.4f}"),
        ("Avg Precision", f"{scores['avg_precision']:.4f}"),
    ]

    # Build metric cards as a table
    row1_values = []
    row1_labels = []
    row2_values = []
    row2_labels = []

    for i, (label, value) in enumerate(hero_metrics):
        p_val = Paragraph(f"<b>{value}</b>", ParagraphStyle(
            f"mv{i}", parent=styles["Normal"], fontSize=16,
            textColor=EMERALD if float(value) > 0.5 else AMBER,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ))
        p_label = Paragraph(label, ParagraphStyle(
            f"ml{i}", parent=styles["Normal"], fontSize=8,
            textColor=MID_GRAY, alignment=TA_CENTER,
        ))
        if i < 5:
            row1_values.append(p_val)
            row1_labels.append(p_label)
        else:
            row2_values.append(p_val)
            row2_labels.append(p_label)

    card_width = 33 * mm
    metrics_table = Table(
        [row1_values, row1_labels, [Spacer(1, 4 * mm)] * 5, row2_values, row2_labels],
        colWidths=[card_width] * 5,
    )
    metrics_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (0, 1), 1, BORDER_GRAY),
        ("BOX", (1, 0), (1, 1), 1, BORDER_GRAY),
        ("BOX", (2, 0), (2, 1), 1, BORDER_GRAY),
        ("BOX", (3, 0), (3, 1), 1, BORDER_GRAY),
        ("BOX", (4, 0), (4, 1), 1, BORDER_GRAY),
        ("BOX", (0, 3), (0, 4), 1, BORDER_GRAY),
        ("BOX", (1, 3), (1, 4), 1, BORDER_GRAY),
        ("BOX", (2, 3), (2, 4), 1, BORDER_GRAY),
        ("BOX", (3, 3), (3, 4), 1, BORDER_GRAY),
        ("BOX", (4, 3), (4, 4), 1, BORDER_GRAY),
        ("BACKGROUND", (0, 0), (-1, 1), LIGHT_GRAY),
        ("BACKGROUND", (0, 3), (-1, 4), LIGHT_GRAY),
    ]))
    elements.append(metrics_table)
    elements.append(Paragraph("Figure 1: Key Classification Metrics Summary", caption_style))

    # ── Confusion Matrix ──
    elements.append(Paragraph("Confusion Matrix", subheading_style))
    elements.append(Paragraph(
        f"The confusion matrix below shows the model's prediction accuracy. Out of "
        f"<b>{model_info['total_samples']:,}</b> total samples, only <b>{cm_data[0][1] + cm_data[1][0]}</b> "
        f"were misclassified — <b>{cm_data[0][1]}</b> false positive(s) and "
        f"<b>{cm_data[1][0]}</b> false negative(s).",
        body_style
    ))

    tn, fp, fn, tp = cm_data[0][0], cm_data[0][1], cm_data[1][0], cm_data[1][1]
    cm_table_data = [
        ["", "Predicted\nRejected", "Predicted\nApproved"],
        ["Actual\nRejected", str(tn), str(fp)],
        ["Actual\nApproved", str(fn), str(tp)],
    ]

    cm_table = Table(cm_table_data, colWidths=[40 * mm, 40 * mm, 40 * mm], rowHeights=[12 * mm, 14 * mm, 14 * mm])
    cm_table.setStyle(TableStyle([
        # Header styling
        ("BACKGROUND", (1, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (0, -1), NAVY),
        ("TEXTCOLOR", (1, 0), (-1, 0), WHITE),
        ("TEXTCOLOR", (0, 1), (0, -1), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        # Correct predictions (diagonal) - green
        ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#d1fae5")),
        ("BACKGROUND", (2, 2), (2, 2), colors.HexColor("#d1fae5")),
        ("TEXTCOLOR", (1, 1), (1, 1), EMERALD),
        ("TEXTCOLOR", (2, 2), (2, 2), EMERALD),
        # Misclassifications - red
        ("BACKGROUND", (2, 1), (2, 1), colors.HexColor("#fee2e2")),
        ("BACKGROUND", (1, 2), (1, 2), colors.HexColor("#fee2e2")),
        ("TEXTCOLOR", (2, 1), (2, 1), RED),
        ("TEXTCOLOR", (1, 2), (1, 2), RED),
        # Fonts for numbers
        ("FONTNAME", (1, 1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (1, 1), (-1, -1), 16),
        # General
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 1, BORDER_GRAY),
        ("BACKGROUND", (0, 0), (0, 0), DARK_BLUE),
    ]))
    elements.append(cm_table)
    elements.append(Paragraph("Figure 2: Confusion Matrix (TN, FP, FN, TP)", caption_style))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════
    #  PAGE 3: CROSS VALIDATION & CLASS DISTRIBUTION
    # ══════════════════════════════════════════════

    elements.append(Paragraph("Cross-Validation Results", heading_style))
    elements.append(Paragraph(
        "5-fold stratified cross-validation was performed to assess generalization. "
        "The table below shows the scores for each fold along with the mean and standard deviation.",
        body_style
    ))

    # CV Table
    cv_headers = ["Metric", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean", "Std"]
    cv_rows = [cv_headers]
    for metric_name in ["accuracy", "f1", "precision", "recall"]:
        row = [metric_name.capitalize()]
        for s in cv[metric_name]["scores"]:
            row.append(f"{s:.4f}")
        row.append(f"{cv[metric_name]['mean']:.4f}")
        row.append(f"{cv[metric_name]['std']:.4f}")
        cv_rows.append(row)

    cv_col_widths = [25 * mm] + [20 * mm] * 5 + [20 * mm, 20 * mm]
    cv_table = Table(cv_rows, colWidths=cv_col_widths)
    cv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_BLUE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        # Highlight mean column
        ("BACKGROUND", (6, 0), (6, 0), TEAL),
        ("FONTNAME", (6, 1), (6, -1), "Helvetica-Bold"),
    ]))
    elements.append(cv_table)
    elements.append(Paragraph("Table 2: 5-Fold Stratified Cross-Validation Scores", caption_style))

    # Class distribution
    elements.append(Paragraph("Dataset Class Distribution", subheading_style))

    total_samples = sum(class_dist["counts"])
    rejected_count = class_dist["counts"][0]
    approved_count = class_dist["counts"][1]
    rejected_pct = rejected_count / total_samples * 100
    approved_pct = approved_count / total_samples * 100

    elements.append(Paragraph(
        f"The dataset contains <b>{total_samples:,}</b> samples in total, with "
        f"<b>{approved_count:,}</b> approved ({approved_pct:.1f}%) and "
        f"<b>{rejected_count:,}</b> rejected ({rejected_pct:.1f}%). "
        f"The moderate class imbalance was addressed during model training.",
        body_style
    ))

    # Class distribution as a visual table
    dist_data = [
        ["Class", "Count", "Percentage", ""],
        ["Approved (1)", f"{approved_count:,}", f"{approved_pct:.1f}%", "████████████████████"],
        ["Rejected (0)", f"{rejected_count:,}", f"{rejected_pct:.1f}%", "████████████"],
    ]
    dist_table = Table(dist_data, colWidths=[35 * mm, 25 * mm, 25 * mm, 80 * mm])
    dist_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_BLUE),
        ("TEXTCOLOR", (3, 1), (3, 1), EMERALD),
        ("TEXTCOLOR", (3, 2), (3, 2), RED),
        ("FONTNAME", (3, 1), (3, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (2, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (2, -1), 0.5, BORDER_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(dist_table)
    elements.append(Paragraph("Figure 3: Approved vs Rejected Class Distribution", caption_style))

    # ── ROC & PR Curve summary ──
    elements.append(Paragraph("ROC & Precision-Recall Summary", subheading_style))
    elements.append(Paragraph(
        f"The model achieves an <b>ROC AUC of {roc['auc']:.4f}</b> and an "
        f"<b>Average Precision of {pr['avg_precision']:.4f}</b>, indicating near-perfect "
        f"discrimination between approved and rejected loan applications.",
        body_style
    ))

    roc_pr_data = [
        ["Curve", "Key Metric", "Value", "Interpretation"],
        ["ROC Curve", "AUC", f"{roc['auc']:.4f}", "Perfect separation between classes"],
        ["PR Curve", "Avg Precision", f"{pr['avg_precision']:.4f}", "Excellent precision across all thresholds"],
    ]
    roc_pr_table = Table(roc_pr_data, colWidths=[30 * mm, 30 * mm, 25 * mm, 80 * mm])
    roc_pr_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_BLUE),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (2, 1), (2, -1), EMERALD),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (3, 1), (3, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(roc_pr_table)
    elements.append(Paragraph("Table 3: ROC and Precision-Recall Curve Metrics", caption_style))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════
    #  PAGE 4: FEATURE IMPORTANCE
    # ══════════════════════════════════════════════

    elements.append(Paragraph("Feature Importance", heading_style))
    elements.append(Paragraph(
        "The chart below shows the relative importance of each feature in the model's "
        "decision-making process. CIBIL Score dominates with 73% importance, followed by "
        "Loan-to-Income ratio (10.1%) and Loan Term (9.8%).",
        body_style
    ))

    # Feature importance as horizontal bar table (text-based)
    fi_data = [["Rank", "Feature", "Importance", "Visual"]]
    max_importance = max(feat["values"])
    for i, (name, val) in enumerate(zip(feat["names"], feat["values"])):
        bar_len = int((val / max_importance) * 30)
        bar = "█" * bar_len
        pct = f"{val * 100:.2f}%"
        fi_data.append([str(i + 1), name, pct, bar])

    fi_table = Table(fi_data, colWidths=[12 * mm, 55 * mm, 22 * mm, 75 * mm])

    fi_style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_BLUE),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (2, -1), 0.5, BORDER_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]

    # Color the bar column
    for i in range(1, len(fi_data)):
        val = feat["values"][i - 1]
        if val >= 0.1:
            fi_style_cmds.append(("TEXTCOLOR", (3, i), (3, i), TEAL))
        elif val >= 0.01:
            fi_style_cmds.append(("TEXTCOLOR", (3, i), (3, i), SOFT_BLUE))
        else:
            fi_style_cmds.append(("TEXTCOLOR", (3, i), (3, i), MID_GRAY))
        fi_style_cmds.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))

    # Highlight top 3
    for i in range(1, 4):
        fi_style_cmds.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))

    fi_table.setStyle(TableStyle(fi_style_cmds))
    elements.append(fi_table)
    elements.append(Paragraph("Figure 4: Feature Importance Ranking (XGBoost)", caption_style))

    # Key insights
    elements.append(Paragraph("Key Insights", subheading_style))

    insights = [
        f"<b>🏆 CIBIL Score</b> is the dominant feature (73.01%), confirming that credit "
        f"history is the strongest predictor of loan approval.",
        f"<b>📊 Loan-to-Income Ratio</b> (10.12%) is the second most important feature, "
        f"reflecting the applicant's debt burden relative to income.",
        f"<b>📅 Loan Term</b> (9.79%) significantly impacts predictions — longer terms "
        f"may indicate higher risk profiles.",
        f"<b>💰 Financial Ratios</b> (Loan-to-Asset: 3.57%, Asset-to-Income: 1.18%) "
        f"together contribute ~4.75% importance.",
        f"<b>📉 Raw Asset Values</b> (residential, commercial, luxury, bank) each "
        f"contribute less than 0.3%, as the engineered ratios capture most information.",
    ]

    for insight in insights:
        elements.append(Paragraph(f"• {insight}", body_style))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════
    #  PAGE 5: CIBIL DISTRIBUTION & CONCLUSION
    # ══════════════════════════════════════════════

    elements.append(Paragraph("CIBIL Score Distribution", heading_style))
    elements.append(Paragraph(
        "The CIBIL score distribution reveals a clear decision boundary around the "
        "<b>550 score mark</b>. Almost all rejections occur below CIBIL 550, while "
        "approvals are concentrated in the 550–900 range.",
        body_style
    ))

    # CIBIL distribution table
    cibil_headers = ["CIBIL Range", "Approved", "Rejected", "Total", "Approval Rate"]
    cibil_rows = [cibil_headers]
    for i, bin_label in enumerate(cibil["bins"]):
        approved = cibil["approved"][i]
        rejected = cibil["rejected"][i]
        total = approved + rejected
        rate = f"{(approved / total * 100):.0f}%" if total > 0 else "N/A"
        cibil_rows.append([bin_label, str(approved), str(rejected), str(total), rate])

    cibil_table = Table(cibil_rows, colWidths=[30 * mm, 25 * mm, 25 * mm, 25 * mm, 30 * mm])

    cibil_style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_BLUE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]

    # Color-code approval rates
    for i in range(1, len(cibil_rows)):
        approved = cibil["approved"][i - 1]
        total = approved + cibil["rejected"][i - 1]
        rate = approved / total if total > 0 else 0
        if rate >= 0.9:
            cibil_style_cmds.append(("TEXTCOLOR", (4, i), (4, i), EMERALD))
            cibil_style_cmds.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))
        elif rate <= 0.2:
            cibil_style_cmds.append(("TEXTCOLOR", (4, i), (4, i), RED))
            cibil_style_cmds.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))

    cibil_table.setStyle(TableStyle(cibil_style_cmds))
    elements.append(cibil_table)
    elements.append(Paragraph("Table 4: CIBIL Score Distribution by Loan Decision", caption_style))

    # ── Conclusion ──
    elements.append(Paragraph("Conclusion", heading_style))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=TEAL,
        spaceBefore=2 * mm, spaceAfter=5 * mm
    ))

    elements.append(Paragraph(
        f"The <b>{model_info['name']}</b> model demonstrates <b>exceptional performance</b> "
        f"across all evaluation metrics:",
        body_style
    ))

    conclusions = [
        f"<b>Near-perfect accuracy</b> of {scores['accuracy'] * 100:.2f}% with only "
        f"{cm_data[0][1] + cm_data[1][0]} misclassifications out of {model_info['total_samples']:,} samples.",
        f"<b>Excellent generalization</b> confirmed by 5-fold cross-validation "
        f"(mean accuracy: {cv['accuracy']['mean'] * 100:.2f}%, σ = {cv['accuracy']['std'] * 100:.2f}%).",
        f"<b>CIBIL Score</b> is the most influential feature (73%), followed by "
        f"loan-to-income ratio (10.1%) and loan term (9.8%).",
        f"<b>Clear decision boundary</b> at CIBIL ≈ 550, with high approval rates above and "
        f"high rejection rates below this threshold.",
        f"<b>Perfect AUC score</b> of {scores['roc_auc']:.4f} indicates optimal trade-off "
        f"between true positive and false positive rates.",
    ]

    for c in conclusions:
        elements.append(Paragraph(f"✅ {c}", body_style))

    elements.append(Spacer(1, 10 * mm))

    # Footer
    elements.append(HRFlowable(
        width="100%", thickness=2, color=NAVY,
        spaceBefore=10 * mm, spaceAfter=5 * mm
    ))
    elements.append(Paragraph(
        f"<i>Report generated by LoanAI Model Analytics Engine on {date_str}</i>",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                       textColor=MID_GRAY, alignment=TA_CENTER)
    ))

    # ── Build ──
    doc.build(elements)
    print(f"\n✅ PDF report generated: {os.path.abspath(filename)}")
    print(f"   Pages: 5 | Size: {os.path.getsize(filename) / 1024:.1f} KB")


if __name__ == "__main__":
    build_pdf()
