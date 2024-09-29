import json 
import streamlit as st

from src.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from src.utils import all_images, all_videos, encode_image, encode_video

@st.cache_data
def cached_encode_image(image):
    return encode_image(image)

@st.cache_data
def cached_encode_video(video):
    return encode_video(video)

def file_upload():
    # Sidebar header
    st.sidebar.header("Upload Files")
    uploaded_files, file_objects = None, None

    uploaded_files = st.sidebar.file_uploader(
        "Please select either up to 3 images or a single video file...",
        type=IMAGE_EXTENSIONS + VIDEO_EXTENSIONS,
        accept_multiple_files=True,
    )

    if uploaded_files is not None and len(uploaded_files) > 0:
        if all_images(uploaded_files) and len(uploaded_files) <= 3:
            with st.sidebar.status("Processing image..."):
                file_objects = [cached_encode_image(image) for image in uploaded_files]
                st.sidebar.image(uploaded_files, use_column_width=True)

        elif all_videos(uploaded_files) and len(uploaded_files) == 1:
            with st.sidebar.status("Processing video..."):
                file_objects = cached_encode_video(uploaded_files[0])
                st.sidebar.video(uploaded_files[0])
        else:
            st.error(
                "Please upload up to 3 images or a single video file with the following extensions: "
                + ", ".join(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS)
            )

    return uploaded_files, file_objects

def header():
    # Inject CSS for centering the image
    center_image_css = """
    <style>
    .centered {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 200px; /* Adjusted size for a smaller logo */
    }
    </style>
    """

    st.markdown(center_image_css, unsafe_allow_html=True)

    # Update the logo image URL here
    st.markdown(
        """
        <img src="https://upload.wikimedia.org/wikipedia/commons/6/67/Logo_Global_NTT_DATA_Future_Blue_RGB.png" class="centered">
        """,
        unsafe_allow_html=True,
    )

    # Change the title and subtitle
    st.markdown(
        "<h1 style='text-align: center;'>Monocular 3D Object Detection and Tracking in Robotic Systems</h1>",
        unsafe_allow_html=True,
    )
    # st.markdown(
    #     "<h2 style='text-align: center;'>Chat with Phi 3.5 vision model</h2>",
    #     unsafe_allow_html=True,
    # )
    # st.markdown(
    #     "<div style='text-align: center; margin-bottom:4'>"
    #     "<p>Phi-3.5-vision is a lightweight, state-of-the-art open multimodal model. <a href='https://huggingface.co/microsoft/Phi-3.5-vision-instruct' target='_blank'>Read more</a></p>"
    #     "</div>",
    #     unsafe_allow_html=True,
    # )

# Advanced settings
def validate_json(input_text):
    try:
        json.loads(input_text)
    except json.JSONDecodeError:
        st.sidebar.warning("Invalid JSON format. Please correct the input.")
        pass

def advanced_settings():
    # This will still handle the logic but won't display anything in the sidebar
    json_mode = st.sidebar.checkbox("Show Advanced Settings", value=False)

    if json_mode:
        st.sidebar.header("Advanced Settings")
        st.sidebar.markdown(
            "[Guide on Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs/introduction)"
        )

        if "schema" not in st.session_state:
            st.session_state.schema = '{"type": "json_object"}'

        def reformat_json():
            try:
                json_str = st.session_state.schema
                json_object = json.loads(json_str)
                formatted_json = json.dumps(json_object, indent=2)
                st.session_state.schema = formatted_json
            except json.JSONDecodeError:
                pass

        schema = st.sidebar.text_area(
            "Edit Schema",
            value=st.session_state.schema,
            help="Define the schema for structured output",
            on_change=reformat_json,
        )
        validate_json(schema)
        return json.loads(schema)
    return None
