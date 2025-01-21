@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure an image file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        # Read the uploaded file
        file = request.files['file']
        try:
            image = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            print(f"Error loading image: {e}")
            return jsonify({"error": "Invalid image file"}), 400

        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt")
        
        # Perform prediction
        outputs = model(**inputs)
        logits = outputs.logits
        print(f"Logits: {logits}")  # Debug: Print logits
        predicted_class_idx = logits.argmax(-1).item()
        print(f"Predicted Index: {predicted_class_idx}")  # Debug: Print predicted index

        # Map predicted index to label
        if 0 <= predicted_class_idx < len(class_labels):
            predicted_label = class_labels[predicted_class_idx]
        else:
            predicted_label = "Unknown"
        print(f"Predicted Label: {predicted_label}")  # Debug: Print predicted label

        # Return prediction
        response = {
            "predicted_class_index": predicted_class_idx,
            "predicted_label": predicted_label
        }
        print(f"Response: {response}")  # Debug: Print response before returning
        return jsonify(response)  # Ensure JSON formatting
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
