from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=int(request.form.get('age')),
            gender=request.form.get('gender'),
            marital_status=request.form.get('marital_status'),
            dependents=int(request.form.get('dependents')),
            education=request.form.get('education'),
            employment_type=request.form.get('employment_type'),
            city_tier=request.form.get('city_tier'),
            residence_type=request.form.get('residence_type'),
            monthly_income=float(request.form.get('monthly_income')),
            cibil_score=int(request.form.get('cibil_score')),
            existing_loans_count=int(request.form.get('existing_loans_count')),
            credit_history_length_years=float(request.form.get('credit_history_length_years')),
            bank_balance=float(request.form.get('bank_balance')),
            loan_amount=float(request.form.get('loan_amount')),
            loan_term_months=int(request.form.get('loan_term_months')),
            loan_purpose=request.form.get('loan_purpose'),
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        pred_label, pred_proba = predict_pipeline.predict(pred_df)

        result = "Approved" if pred_label[0] == 1 else "Rejected"
        confidence = round(float(pred_proba[0]) * 100, 1) if pred_label[0] == 1 else round((1 - float(pred_proba[0])) * 100, 1)

        return render_template('home.html', results=result, confidence=confidence)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
