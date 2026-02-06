import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# Implements softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# use a session state variable to allow for the canvas to be cleared/reset when a button is pressed
if "clear_trigger" not in st.session_state:
    st.session_state.clear_trigger = 0

# add the app title
st.title("Digit Recognizer")

# add the button to clear the canvas
if st.button('Clear'):
    st.session_state.clear_trigger += 1

# add a drawable canvas object to the app where a user can draw a digit
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
    stroke_width=25,
    stroke_color="#000000",
    background_color="#ffffff",
    width=256,
    height=256,
    drawing_mode="freedraw",
    # using a dynamically changing key is what allows the canvas to be cleared
    key=f"canvas_{st.session_state.clear_trigger}",
)

# add some placeholder text for app instructions or the results of the digit classification
message_text = st.empty()
message_text.text('Scribble a digit and press upload.')

# add the upload button to trigger when the digit analysis should start
if st.button('Upload'):
    if canvas_result.image_data is not None:

        # Load Digit Recognition model
        net = cv2.dnn.readNetFromONNX('model.onnx')

        # Create a 4D blob from the inverted image from the canvas
        blob = cv2.dnn.blobFromImage(cv2.bitwise_not(canvas_result.image_data), 1 / 255, (28, 28))

        # Run a model
        net.setInput(blob)
        out = net.forward()

        # Get a class with a highest score
        out = softmax(out.flatten())
        classId = np.argmax(out)
        confidence = out[classId]

        # output the detected digit and confidence back to the app
        message_text.text(f"classId: {classId} confidence: {confidence}")
