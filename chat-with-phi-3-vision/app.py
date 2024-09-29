import json
import socket
import streamlit as st
import numpy as np
import cv2
import base64
from src.api import client
from src.config import SYSTEM_MESSAGE, PHI_VISION_MODELS
from src.ui_components import header, advanced_settings
from src.utils import prepare_content_with_images

def setup_socket_client(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    return client_socket

def receive_image(client_socket):
    # Receive the size of the incoming image
    image_size = int.from_bytes(client_socket.recv(4), byteorder='big')
    # Receive the image data
    image_data = b''
    while len(image_data) < image_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        image_data += packet
    # Decode the image from bytes
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def main():
    # Title section
    header()

    model_option = st.sidebar.selectbox(
        "Select Phi Vision Model",
        PHI_VISION_MODELS.keys(),
        index=0,
    )
    MODEL = PHI_VISION_MODELS[model_option]





    # Advanced Settings
    response_format = advanced_settings()

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []


    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #007BFF; /* Blue color */
            color: white; /* Text color */
            border: none; /* Remove border */
            border-radius: 20px; /* More rounded corners */
            padding: 10px 20px; /* Add some padding */
            transition: background-color 0.3s, transform 0.3s; /* Transition for hover effects */
        }
        .stButton > button:hover {
            background-color: #0056b3; /* Darker blue on hover */
            transform: scale(1.05); /* Slightly enlarge on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "messages" in st.session_state.keys() and len(st.session_state.messages) > 0:
        # Add clear chat history button to sidebar
        st.sidebar.button(
            "Clear Chat History",
            on_click=lambda: st.session_state.messages.clear(),
            type="primary",
        )


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Display chat message in chat message container
        with st.chat_message(message["role"]):
            content = message["content"]
            if isinstance(content, list):
                st.markdown(content[0]["text"])
                urls = [item["image_url"]["url"] for item in content[1:]]
                st.image(urls if len(urls) < 3 else urls[0], width=200)
            else:
                st.markdown(content)





    if prompt := st.chat_input("Ask something", key="prompt"):
        # Display user message in chat message container
        with st.chat_message("user"):
            host = '127.0.0.1'
            port = 8502
            client_socket = setup_socket_client(host, port)
            image = receive_image(client_socket)
            
            client_socket.close()
            st.image(image, channels="BGR", width = 200)
            
            st.markdown(prompt)
            _, buffer = cv2.imencode('.jpg', image)
            img_bytes = buffer.tobytes()
            content = prepare_content_with_images(prompt, [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"} }])
            

            st.session_state.messages.append({"role": "user", "content": content})

        # Get response from the assistant
        with st.chat_message("assistant"):
            messages = [SYSTEM_MESSAGE, *st.session_state.messages]
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
                response_format=response_format,
            )
            response = st.write_stream(stream)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()









