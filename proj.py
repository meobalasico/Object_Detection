import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient
import tempfile

# Define the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VsxreoZsgrCDLK4xweXv"
)

MODEL_ID = "face-detection-mik1i/21"

def infer_image(image):
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name
    
    # Use the inference client to send the image
    result = CLIENT.infer(temp_file_path, model_id=MODEL_ID)
    return result

# Set page title and favicon
st.set_page_config(page_title="Face Detection App", page_icon="👩‍🦰")

# Define app title and subtitle with emojis
st.title("👀 Face Detection App")
st.write(
    "This app detects faces in images using a pre-trained model. Upload an image to get started! 📷"
)

# Add space for better layout
st.write("")

# Add file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Displaying preloader while detecting faces
    with st.spinner("Detecting faces..."):
        result = infer_image(image)

    # Display results after preloader
    if result and "predictions" in result:
        # Draw bounding boxes on the annotated image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        for prediction in result["predictions"]:
            # Extract bounding box coordinates
            x_center = prediction.get("x")
            y_center = prediction.get("y")
            width = prediction.get("width")
            height = prediction.get("height")
            confidence = prediction.get("confidence")
            # Calculate top-left and bottom-right coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            # Draw bounding box and label
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label = f"Face ({confidence:.2f})"
            draw.text((x1, y1), label, fill="red")
                
        # Display annotated image
        st.subheader("🖼️ Annotated Image with Bounding Boxes")
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
        
        # Display number of faces detected
        num_faces = len(result["predictions"])
        st.subheader(f"Number of Faces Detected: {num_faces}")
        
        # Display bounding box results
        st.subheader("🔍 Bounding Box Results")
        for idx, prediction in enumerate(result["predictions"]):
            st.write(f"Face {idx + 1}:")
            st.write(prediction)
        
    else:
        st.markdown('<p style="color: green; font-weight: bold;">No faces detected. 😔</p>', unsafe_allow_html=True)