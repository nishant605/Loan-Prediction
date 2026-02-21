import os
import io
import pickle
import uuid
import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, session

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'loanai-secret-key-2026'

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    logger.info('Model loaded successfully')
except Exception as e:
    logger.error(f'Failed to load model: {e}')
    raise

# In-memory storage for batch results (keyed by session ID)
batch_results = {}

# Required columns for batch prediction
REQUIRED_COLS = [
    'no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]

OPTIONAL_COLS = ['total_assets_value', 'loan_to_income', 'loan_to_asset', 'asset_to_income']


def add_derived_features(df):
    """Add engineered features if they don't exist."""
    df = df.copy()

    if 'total_assets_value' not in df.columns:
        df['total_assets_value'] = (
            df['residential_assets_value'] +
            df['commercial_assets_value'] +
            df['luxury_assets_value'] +
            df['bank_asset_value']
        )

    if 'loan_to_income' not in df.columns:
        df['loan_to_income'] = df['loan_amount'] / df['income_annum'].replace(0, 1)

    if 'loan_to_asset' not in df.columns:
        df['loan_to_asset'] = df['loan_amount'] / df['total_assets_value'].replace(0, 1)

    if 'asset_to_income' not in df.columns:
        df['asset_to_income'] = df['total_assets_value'] / df['income_annum'].replace(0, 1)

    return df


# ---- Page Routes ----

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/performance')
def performance():
    return render_template('performance.html')


@app.route('/batch')
def batch():
    return render_template('batch.html')


# ---- Single Predict API ----

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        dependents = int(data['dependents'])
        education = data['education']
        self_employed = data['selfEmployed']
        income = float(data['income'])
        loan_amount = float(data['loanAmount'])
        loan_term = int(data['loanTerm'])
        cibil_score = int(data['cibilScore'])
        total_assets = float(data['totalAssets'])

        residential = total_assets * 0.25
        commercial = total_assets * 0.25
        luxury = total_assets * 0.25
        bank = total_assets * 0.25

        loan_to_income = loan_amount / income if income > 0 else 0
        loan_to_asset = loan_amount / total_assets if total_assets > 0 else 0
        asset_to_income = total_assets / income if income > 0 else 0

        input_data = pd.DataFrame([{
            "no_of_dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential,
            "commercial_assets_value": commercial,
            "luxury_assets_value": luxury,
            "bank_asset_value": bank,
            "total_assets_value": total_assets,
            "loan_to_income": loan_to_income,
            "loan_to_asset": loan_to_asset,
            "asset_to_income": asset_to_income
        }])

        prediction = model.predict(input_data)

        R = 8 / (12 * 100)
        N = loan_term * 12
        emi = (loan_amount * R * (1 + R) ** N) / ((1 + R) ** N - 1) if R > 0 and N > 0 else 0

        result = {
            'approved': bool(prediction[0] == 1),
            'summary': {
                'annualIncome': income,
                'loanAmount': loan_amount,
                'loanTerm': loan_term,
                'totalAssets': total_assets,
                'emi': round(emi, 2),
                'loanToIncome': round(loan_to_income, 2),
                'loanToAsset': round(loan_to_asset * 100, 1),
                'assetToIncome': round(asset_to_income, 2),
                'cibilScore': cibil_score,
                'dependents': dependents,
                'education': education,
                'selfEmployed': self_employed,
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ---- Batch Predict API ----

@app.route('/batch/upload', methods=['POST'])
def batch_upload():
    """Upload CSV/Excel, validate columns, run predictions, return JSON preview."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        filename = file.filename.lower()

        if not filename:
            return jsonify({'error': 'No file selected'}), 400

        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload CSV or Excel (.xlsx/.xls).'}), 400

        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Strip whitespace from string columns
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

        # Check required columns
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing)}',
                'required_columns': REQUIRED_COLS,
                'found_columns': df.columns.tolist()
            }), 400

        # Drop rows with NaN in required columns
        original_count = len(df)
        df = df.dropna(subset=REQUIRED_COLS)
        dropped = original_count - len(df)

        if len(df) == 0:
            return jsonify({'error': 'No valid rows found after removing missing values.'}), 400

        # Add derived features
        pred_df = add_derived_features(df[REQUIRED_COLS])

        # Predict
        feature_cols = REQUIRED_COLS + OPTIONAL_COLS
        predictions = model.predict(pred_df[feature_cols])
        probabilities = model.predict_proba(pred_df[feature_cols])

        # Calculate EMI for each row
        R = 8 / (12 * 100)
        emis = []
        for _, row in pred_df.iterrows():
            N = int(row['loan_term']) * 12
            if R > 0 and N > 0:
                emi = (row['loan_amount'] * R * (1 + R) ** N) / ((1 + R) ** N - 1)
            else:
                emi = 0
            emis.append(round(emi, 2))

        # Build result DataFrame
        result_df = df[REQUIRED_COLS].copy()
        result_df['prediction'] = ['Approved' if p == 1 else 'Rejected' for p in predictions]
        result_df['confidence'] = [round(max(p) * 100, 1) for p in probabilities]
        result_df['estimated_emi'] = emis

        # Store in memory with a unique batch ID
        batch_id = str(uuid.uuid4())[:8]
        batch_results[batch_id] = {
            'dataframe': result_df,
            'filename': file.filename,
            'total_rows': len(result_df),
            'approved_count': int(sum(predictions == 1)),
            'rejected_count': int(sum(predictions == 0)),
        }

        # Build JSON preview (first 50 rows)
        preview = result_df.head(50).to_dict(orient='records')

        return jsonify({
            'batch_id': batch_id,
            'filename': file.filename,
            'total_rows': len(result_df),
            'dropped_rows': dropped,
            'approved_count': int(sum(predictions == 1)),
            'rejected_count': int(sum(predictions == 0)),
            'avg_confidence': round(result_df['confidence'].mean(), 1),
            'columns': result_df.columns.tolist(),
            'preview': preview,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/batch/download/excel/<batch_id>')
def batch_download_excel(batch_id):
    """Download batch results as Excel."""
    if batch_id not in batch_results:
        return jsonify({'error': 'Batch not found. Please run a new prediction.'}), 404

    data = batch_results[batch_id]
    df = data['dataframe']

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')

        # Auto-adjust column widths
        ws = writer.sheets['Predictions']
        for i, col in enumerate(df.columns, 1):
            max_len = max(len(str(col)), df[col].astype(str).str.len().max())
            ws.column_dimensions[chr(64 + i) if i <= 26 else 'A' + chr(64 + i - 26)].width = min(max_len + 3, 30)

    output.seek(0)

    fname = f"LoanAI_Batch_Predictions_{batch_id}.xlsx"
    return send_file(output, download_name=fname, as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.route('/batch/download/pdf/<batch_id>')
def batch_download_pdf(batch_id):
    """Download batch results as PDF."""
    if batch_id not in batch_results:
        return jsonify({'error': 'Batch not found. Please run a new prediction.'}), 404

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    data = batch_results[batch_id]
    df = data['dataframe']

    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(A4),
                            leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=20 * mm, bottomMargin=15 * mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                 fontSize=18, textColor=colors.HexColor('#1e3a5f'),
                                 spaceAfter=6 * mm)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                    fontSize=10, textColor=colors.HexColor('#6b7280'),
                                    spaceAfter=8 * mm)

    elements = []

    # Title
    elements.append(Paragraph("LoanAI — Batch Prediction Report", title_style))
    elements.append(Paragraph(
        f"File: {data['filename']} &nbsp;|&nbsp; "
        f"Total: {data['total_rows']} rows &nbsp;|&nbsp; "
        f"Approved: {data['approved_count']} &nbsp;|&nbsp; "
        f"Rejected: {data['rejected_count']}",
        subtitle_style
    ))

    # Simplify columns for PDF readability
    pdf_cols = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
                'loan_amount', 'loan_term', 'cibil_score', 'prediction', 'confidence', 'estimated_emi']
    pdf_df = df[[c for c in pdf_cols if c in df.columns]]

    # Header labels (shortened for PDF)
    header_map = {
        'no_of_dependents': 'Deps',
        'education': 'Education',
        'self_employed': 'Self Emp',
        'income_annum': 'Income',
        'loan_amount': 'Loan Amt',
        'loan_term': 'Term',
        'cibil_score': 'CIBIL',
        'prediction': 'Result',
        'confidence': 'Conf %',
        'estimated_emi': 'EMI',
    }

    headers = [header_map.get(c, c) for c in pdf_df.columns]
    table_data = [headers]

    # Format values for PDF
    for _, row in pdf_df.iterrows():
        formatted_row = []
        for col in pdf_df.columns:
            val = row[col]
            if col in ('income_annum', 'loan_amount', 'estimated_emi'):
                formatted_row.append(f"₹{val:,.0f}" if pd.notna(val) else '-')
            elif col == 'confidence':
                formatted_row.append(f"{val}%")
            else:
                formatted_row.append(str(val))
        table_data.append(formatted_row)

    # Build table
    col_widths = [30, 60, 45, 65, 65, 30, 40, 55, 40, 65]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)

    # Styling
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]

    # Color code prediction column
    pred_col_idx = list(pdf_df.columns).index('prediction') if 'prediction' in pdf_df.columns else -1
    if pred_col_idx >= 0:
        for i, row in enumerate(table_data[1:], 1):
            if row[pred_col_idx] == 'Approved':
                style_cmds.append(('TEXTCOLOR', (pred_col_idx, i), (pred_col_idx, i), colors.HexColor('#059669')))
                style_cmds.append(('FONTNAME', (pred_col_idx, i), (pred_col_idx, i), 'Helvetica-Bold'))
            else:
                style_cmds.append(('TEXTCOLOR', (pred_col_idx, i), (pred_col_idx, i), colors.HexColor('#dc2626')))
                style_cmds.append(('FONTNAME', (pred_col_idx, i), (pred_col_idx, i), 'Helvetica-Bold'))

    t.setStyle(TableStyle(style_cmds))
    elements.append(t)

    doc.build(elements)
    output.seek(0)

    fname = f"LoanAI_Batch_Report_{batch_id}.pdf"
    return send_file(output, download_name=fname, as_attachment=True, mimetype='application/pdf')


@app.route('/batch/sample')
def batch_sample():
    """Download a sample template CSV."""
    sample = pd.DataFrame([
        {
            'no_of_dependents': 2, 'education': 'Graduate', 'self_employed': 'No',
            'income_annum': 9600000, 'loan_amount': 29900000, 'loan_term': 12,
            'cibil_score': 778, 'residential_assets_value': 2400000,
            'commercial_assets_value': 17600000, 'luxury_assets_value': 22700000,
            'bank_asset_value': 8000000
        },
        {
            'no_of_dependents': 0, 'education': 'Not Graduate', 'self_employed': 'Yes',
            'income_annum': 4100000, 'loan_amount': 12200000, 'loan_term': 8,
            'cibil_score': 417, 'residential_assets_value': 2700000,
            'commercial_assets_value': 2200000, 'luxury_assets_value': 8800000,
            'bank_asset_value': 3300000
        },
        {
            'no_of_dependents': 1, 'education': 'Graduate', 'self_employed': 'No',
            'income_annum': 7500000, 'loan_amount': 15000000, 'loan_term': 15,
            'cibil_score': 720, 'residential_assets_value': 5000000,
            'commercial_assets_value': 4000000, 'luxury_assets_value': 9000000,
            'bank_asset_value': 3500000
        },
    ])

    output = io.BytesIO()
    sample.to_csv(output, index=False)
    output.seek(0)

    return send_file(output, download_name='LoanAI_Sample_Template.csv',
                     as_attachment=True, mimetype='text/csv')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
