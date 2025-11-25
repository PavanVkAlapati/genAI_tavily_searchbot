# streamlit_client.py

import os
import uuid
import requests
import streamlit as st
from io import BytesIO

# Try to import reportlab for PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None  # PDF export will be disabled if reportlab is missing


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

API_URL = os.getenv("NEWS_AGENT_API_URL", "http://127.0.0.1:8000/chat")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
USER_ICON_PATH = os.path.join(ASSETS_DIR, "user.png")
BOT_ICON_PATH = os.path.join(ASSETS_DIR, "bot.png")
SEND_ICON_PATH = os.path.join(ASSETS_DIR, "send.png")
DOWNLOAD_ICON_PATH = os.path.join(ASSETS_DIR, "download.png")

st.set_page_config(
    page_title="SOTIN",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "⚡",
    layout="wide",
)


# -------------------------------------------------------------------
# SOTIN Purple Glass Theme — CSS
# -------------------------------------------------------------------

SOTIN_CSS = """
<style>
:root {
    --sotin-bg: #05030d;
    --sotin-panel: #0b0618;
    --sotin-panel-soft: #120b24;
    --sotin-purple: #a55bff;
    --sotin-purple-dark: #5a1bb3;
    --sotin-purple-soft: rgba(165, 91, 255, 0.12);
    --sotin-text: #f7f1ff;
    --sotin-muted: #b5a3e6;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #201040 0%, #05030d 40%, #000000 100%);
    color: var(--sotin-text);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090215, #05030d);
    border-right: 1px solid rgba(165, 91, 255, 0.4);
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--sotin-panel-soft);
    border-radius: 18px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.4rem;
    box-shadow: 0 0 18px rgba(0,0,0,0.55);
}

/* Chat input bubble */
.sotin-input-row {
    padding: 0.6rem 0.3rem 0.8rem 0.3rem;
    border-radius: 999px;
    background: radial-gradient(circle at top left,
        rgba(165, 91, 255, 0.22),
        rgba(10, 6, 25, 0.98));
    box-shadow: 0 0 22px rgba(165, 91, 255, 0.4);
}

.sotin-input-row input {
    background: transparent !important;
    border: none !important;
    color: var(--sotin-text) !important;
}

[data-baseweb="input"] > div {
    background: transparent !important;
    border: none !important;
}

/* Suggestion tiles */
.sotin-suggestions-header {
    font-size: 0.85rem;
    color: var(--sotin-muted);
    margin-top: 0.3rem;
    margin-bottom: 0.3rem;
}

.sotin-suggestion button {
    width: 100%;
    text-align: left;
    font-size: 0.80rem;
    white-space: normal;
    background: linear-gradient(135deg, var(--sotin-purple), var(--sotin-purple-dark)) !important;
    border-radius: 14px !important;
    box-shadow: 0 0 14px rgba(165, 91, 255, 0.35);
}

/* Buttons */
button[kind="primary"],
button[kind="secondary"] {
    background: linear-gradient(135deg, var(--sotin-purple), var(--sotin-purple-dark)) !important;
    border-radius: 999px !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: white !important;
}
</style>
"""

st.markdown(SOTIN_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------------
# Session State
# -------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "vk"

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def call_api(uid, sid, msg):
    payload = {"user_id": uid, "session_id": sid, "message": msg}
    r = requests.post(API_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def build_export_markdown(msgs):
    """Builds a markdown-formatted transcript."""
    out = ["# SOTIN — Chat Transcript", ""]
    for m in msgs:
        who = "**User**" if m["role"] == "user" else "**Assistant**"
        out.append(f"{who}:")
        out.append(m["content"])
        if m.get("citations"):
            out.append("")
            out.append("  - **Sources:**")
            for c in m["citations"]:
                title = c.get("title") or ""
                url = c.get("url") or ""
                snippet = c.get("snippet") or ""
                line = f"    - [{title}]({url})"
                if snippet:
                    line += f" — _{snippet}_"
                out.append(line)
        out.append("")
    return "\n".join(out)


def build_export_pdf(markdown_text: str):
    """Converts markdown text to a very simple PDF (plain text)."""
    if canvas is None:
        return None  # reportlab not available

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    x_margin = 40
    y_margin = 40
    y = height - y_margin
    line_height = 14

    # crude text-only rendering: no markdown styling, just plain lines
    for raw_line in markdown_text.split("\n"):
        line = raw_line.replace("\t", "    ")
        if y < y_margin:
            c.showPage()
            y = height - y_margin
        c.drawString(x_margin, y, line[:120])  # trim very long lines
        y -= line_height

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

    st.markdown("#### Session Info")
    st.caption(f"Session ID: `{st.session_state['session_id']}`")

    if st.session_state["messages"]:
        st.markdown("---")
        st.markdown("#### Export Chat")

        if os.path.exists(DOWNLOAD_ICON_PATH):
            st.image(DOWNLOAD_ICON_PATH, width=40)

        md_text = build_export_markdown(st.session_state["messages"])

        # Markdown download
        st.download_button(
            "📄 Download as Markdown (.md)",
            md_text,
            "sotin_chat.md",
            mime="text/markdown",
        )

        # PDF download (if reportlab is available)
        pdf_bytes = build_export_pdf(md_text)
        if pdf_bytes is not None:
            st.download_button(
                "📚 Download as PDF (.pdf)",
                pdf_bytes,
                "sotin_chat.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("PDF export disabled.")

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=70)
with col2:
    st.markdown("<h1>SOTIN Chat</h1>", unsafe_allow_html=True)
    st.caption(
        "Search On Tavily If Needed, A Tavily-powered news & web assistant."
    )

st.markdown("---")


# -------------------------------------------------------------------
# Existing Chat Messages
# -------------------------------------------------------------------

for m in st.session_state["messages"]:
    avatar = USER_ICON_PATH if m["role"] == "user" else BOT_ICON_PATH
    avatar = avatar if os.path.exists(avatar) else None

    with st.chat_message(m["role"], avatar=avatar):
        st.write(m["content"])

        if m.get("citations"):
            with st.expander("Citations"):
                for c in m["citations"]:
                    url = c.get("url")
                    title = c.get("title") or url
                    st.markdown(f"- [{title}]({url})")
                    if c.get("snippet"):
                        st.caption(c["snippet"])


# -------------------------------------------------------------------
# Suggestion Tiles
# -------------------------------------------------------------------

SUGGESTIONS = [
    "Summarize today's top AI stories.",
    "What are the most important world headlines?",
    "Give me a short note of stock market news.",
    "What's new in technology and startups?",
]

clicked_prompt = None

st.markdown(
    "<div class='sotin-suggestions-header'>Try asking »</div>",
    unsafe_allow_html=True,
)

cols = st.columns(4)
for i, (c, text) in enumerate(zip(cols, SUGGESTIONS)):
    with c:
        st.markdown("<div class='sotin-suggestion'>", unsafe_allow_html=True)
        if st.button(text, key=f"sg_{i}"):
            clicked_prompt = text
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Input Box
# -------------------------------------------------------------------

in_col, icon_col = st.columns([14, 1])

with in_col:
    st.markdown('<div class="sotin-input-row">', unsafe_allow_html=True)
    typed_prompt = st.chat_input("Type your message...")
    st.markdown("</div>", unsafe_allow_html=True)

with icon_col:
    if os.path.exists(SEND_ICON_PATH):
        st.image(SEND_ICON_PATH, width=42)

final_prompt = clicked_prompt or typed_prompt


# -------------------------------------------------------------------
# Handle user message
# -------------------------------------------------------------------

if final_prompt:
    st.session_state["messages"].append(
        {"role": "user", "content": final_prompt, "citations": []}
    )

    try:
        res = call_api(
            st.session_state["user_id"],
            st.session_state["session_id"],
            final_prompt,
        )
        answer = res.get("final_answer", "")
        cites = res.get("citations", [])
    except Exception as e:
        answer = f"Error: {e}"
        cites = []

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "citations": cites}
    )

    st.rerun()
