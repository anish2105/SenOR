import streamlit as st
from back import *
#import back
st.set_page_config(page_title="💬👨🏻‍⚖️ SenOR Legal Advisor")
st.title("🤖 SenOR ")
with st.sidebar:
    st.markdown("<h1 style='text-align:center;font-family:Georgia;font-size:26px;'>🧑‍⚖️ SenOR Legal Advisor </h1>",
                unsafe_allow_html=True)
    st.markdown("<h7 style='text-align: justify;font-size:20px;'>This app is a smart legal chatbot that is integrated into an easy-to-use platform. This would give lawyers "
                "instant access to legal information of Women’s Legal Rights and remove the need for laborious manual research in books or regulations using the power "
                "of Large Language Models</h7>", unsafe_allow_html=True)
    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Features</h1>", unsafe_allow_html=True)

    st.markdown(" - Users can adjust token length to control the length of generated responses, allowing for customization based on specific requirements or constraints.")
    st.markdown(" - Users can adjust the temp to control response randomness. Higher values (e.g., 0.5) produce diverse but less focused responses, while low values (e.g., 0.1) result in more focused but less varied answers.")
    st.markdown("- users can now use text to speech")
    st.markdown("<h2 style='font-family:Georgia;font-size:20px;'>Press enter to stop the audio</h1>", unsafe_allow_html=True)

    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>",
                unsafe_allow_html=True)
    max_length = st.slider("Token Max Length", min_value=128, max_value=3000, value=512, step=128)
    temp = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

    language = st.radiobox("Select Language for Speech", options=['en', 'hi', 'es', 'fr', 'de', 'zh-cn'])


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_response(prompt, max_length, temp):
    response = model(prompt, max_length, temp)
    full_response = ''.join(response)
    return full_response

if user_prompt := st.chat_input("Enter your query"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(user_prompt, max_length, temp)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

if st.button("Repeat Last Response"):
    last_user_message = None
    for message in reversed(st.session_state.messages):
        if message["role"] == "user":
            last_user_message = message["content"]
            break
    if last_user_message:
        with st.chat_message("assistant"):
            with st.spinner("Regenerating response..."):
                response = generate_response(last_user_message, max_length, temp)
                placeholder = st.empty()
                placeholder.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

if st.button("Convert to Speech"):
    if st.session_state.messages[-1]["role"] == "assistant":
        audio_base64 = text_speech(st.session_state.messages[-1]["content"], lang=language)
        st.audio(base64.b64decode(audio_base64), format='audio/mp3')

st.markdown("""
    <div style="background-color: #FFFF8F; padding: 10px; border-radius: 5px;">
        <p style="color: black;">⚠️Senor can make mistakes. Information is updated only till 2023. Please consider checking important information.</p>
    </div>
""", unsafe_allow_html=True)
