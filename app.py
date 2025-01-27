from PIL import ImageDraw, ImageFont


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("=== /predict endpoint accessed ===")

        # Ensure the file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        # Read and preprocess the image
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = vit_model(input_tensor)
            predicted_class_idx = outputs.argmax(dim=1).item()
        predicted_label = class_labels[predicted_class_idx]

        # Draw the label on the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", size=24)  # Ensure you have this font available
        draw.text((10, 10), f"{predicted_label}", fill="red", font=font)

        # Save the updated image with predictions
        output_path = "output_image.jpg"
        image.save(output_path)
        print(f"Image saved with predictions: {output_path}")

        return jsonify({
            "predicted_class_index": predicted_class_idx,
            "predicted_label": predicted_label,
            "output_image": output_path
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
