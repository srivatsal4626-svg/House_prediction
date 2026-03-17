@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction = None

    if request.method == 'POST':
        try:
            if model is None or encoder is None:
                flash("Model not loaded properly.", "danger")
                return redirect(url_for('dashboard'))

            area = float(request.form.get('area') or 0)
            bedrooms = int(request.form.get('bedrooms') or 0)
            bathrooms = int(request.form.get('bathrooms') or 0)
            location = request.form.get('location')

            if location not in encoder.classes_:
                flash("Invalid location selected", "danger")
                return redirect(url_for('dashboard'))

            location_encoded = encoder.transform([location])[0]

            features = np.array([[area, bedrooms, bathrooms, location_encoded]])

            predicted_price = model.predict(features)[0]
            prediction = round(predicted_price, 2)

        except Exception as e:
            print(e)  # important for terminal debugging
            flash("Error making prediction. Check inputs.", "danger")

    locations = encoder.classes_ if encoder else []
    return render_template('dashboard.html', prediction=prediction, locations=locations)
