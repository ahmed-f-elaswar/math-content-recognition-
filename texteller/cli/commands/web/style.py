"""HTML style definitions for the TexTeller Streamlit web interface.

This module contains HTML templates and CSS styles used in the Streamlit
web application for TexTeller. It defines visual elements for branding,
user feedback, and content display.

Constants:
    HEADER_HTML (str): HTML for the main header with TexTeller branding including
        logo images and title text.
    SUCCESS_GIF_HTML (str): HTML for success animation showing clapping hands,
        displayed when recognition completes successfully.
    FAIL_GIF_HTML (str): HTML for failure animation showing glitch effect,
        displayed when recognition fails or encounters errors.
    IMAGE_EMBED_HTML (str): HTML template for displaying base64-encoded images
        with CSS for centered container and size constraints. Use .format()
        with img_base64 parameter.
    IMAGE_INFO_HTML (str): HTML template for displaying image dimension information.
        Use .format() with img_height and img_width parameters.

Examples:
    Using the header in Streamlit::
    
        import streamlit as st
        from texteller.cli.commands.web.style import HEADER_HTML
        
        st.markdown(HEADER_HTML, unsafe_allow_html=True)
    
    Displaying an image with info::
    
        import base64
        from texteller.cli.commands.web.style import IMAGE_EMBED_HTML, IMAGE_INFO_HTML
        
        # Encode image
        with open("image.png", "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        # Display image with centered layout
        html = IMAGE_EMBED_HTML.format(img_base64=img_base64)
        st.markdown(html, unsafe_allow_html=True)
        
        # Display dimensions
        info = IMAGE_INFO_HTML.format(img_height=480, img_width=640)
        st.markdown(info, unsafe_allow_html=True)
    
    Showing success animation::
    
        from texteller.cli.commands.web.style import SUCCESS_GIF_HTML
        
        st.markdown(SUCCESS_GIF_HTML, unsafe_allow_html=True)

Notes:
    All HTML strings use the lines_dedent utility to remove leading whitespace
    while preserving the HTML structure. This makes the source code more readable.
"""

from texteller.utils import lines_dedent


HEADER_HTML = lines_dedent("""
    <h1 style="color: black; text-align: center;">
        <img src="https://raw.githubusercontent.com/OleehyO/TexTeller/main/assets/fire.svg" width="100">
        𝚃𝚎𝚡𝚃𝚎𝚕𝚕𝚎𝚛
        <img src="https://raw.githubusercontent.com/OleehyO/TexTeller/main/assets/fire.svg" width="100">
    </h1>
    """)

SUCCESS_GIF_HTML = lines_dedent("""
    <h1 style="color: black; text-align: center;">
        <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
        <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
        <img src="https://slackmojis.com/emojis/90621-clapclap-e/download" width="50">
    </h1>
    """)

FAIL_GIF_HTML = lines_dedent("""
    <h1 style="color: black; text-align: center;">
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download">
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download">
        <img src="https://slackmojis.com/emojis/51439-allthethings_intensifies/download">
    </h1>
    """)

IMAGE_EMBED_HTML = lines_dedent("""
    <style>
    .centered-container {{
        text-align: center;
    }}
    .centered-image {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-height: 350px;
        max-width: 100%;
    }}
    </style>
    <div class="centered-container">
        <img src="data:image/png;base64,{img_base64}" class="centered-image" alt="Input image">
    </div>
    """)

IMAGE_INFO_HTML = lines_dedent("""
    <style>
    .centered-container {{
        text-align: center;
    }}
    </style>
    <div class="centered-container">
        <p style="color:gray;">Input image ({img_height}✖️{img_width})</p>
    </div>
    """)
