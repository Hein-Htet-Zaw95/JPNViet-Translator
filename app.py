import io
import os
import time
import tempfile
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from pydub import AudioSegment

# -----------------------------
# 初期化
# -----------------------------
load_dotenv()
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
    st.warning("OPENAI_API_KEY が設定されていません。 .env に追加してください。")

client = OpenAI()

APP_TITLE = "🇻🇳⇄🇯🇵 ベトナム語 ⇄ 日本語 翻訳 (テキスト + 音声)"
STT_MODEL = "gpt-4o-mini-transcribe"     # 音声→テキスト
TTS_MODEL = "gpt-4o-mini-tts"             # テキスト→音声
LLM_MODEL = "gpt-4o-mini"                 # 翻訳

st.set_page_config(page_title=APP_TITLE, page_icon="🌏", layout="centered")
st.title(APP_TITLE)
st.caption("テキスト翻訳、マイク入力、音声会話。Streamlit + OpenAI で構築。")

# -----------------------------
# ヘルパー関数
# -----------------------------

def detect_lang_simple(text: str) -> str:
    """ベトナム語/日本語の簡易判定"""
    if any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in text):
        return "ja"
    try:
        lang = detect(text)
        if lang in ("ja", "vi"):
            return lang
    except Exception:
        pass
    return "vi" if all(ord(c) < 128 for c in text) else "ja"


def translate_text(text: str, src: str, dst: str) -> str:
    if src == "auto":
        detected = detect_lang_simple(text)
        if detected in ("vi", "ja"):
            src = detected
        else:
            src = "vi"  # default fallback
    if src == dst:
        return text

    system_prompt = (
        "あなたはプロの翻訳者です。文章を簡潔かつ自然に翻訳してください。"
        "- ソース言語: 'vi'=ベトナム語, 'ja'=日本語"
        "- ターゲット言語: 'ja'=日本語, 'vi'=ベトナム語"
        "- 数字や名前はそのまま保持"
        "- 説明は追加せず翻訳文のみ出力"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[SRC={src}] [DST={dst}]\n{text}"},
    ]

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,  # type: ignore
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip() if resp.choices[0].message.content else "Translation failed"
    except Exception as e:
        return f"Translation error: {str(e)}"


def transcribe_bytes(wav_bytes: bytes, lang_hint: str = "auto") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            kwargs = {"model": STT_MODEL, "file": f}
            if lang_hint in ("vi", "ja"):
                kwargs["language"] = lang_hint
            stt = client.audio.transcriptions.create(**kwargs)
        return stt.text.strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def speak(text: str, voice: str = "alloy", fmt: str = "mp3"):
    """TTS（format 引数なし）。必要ならローカルで MP3→WAV 変換。戻り値は (bytes, mime)。"""
    if not text.strip():
        return b"", "audio/mp3"

    # 新SDKでは format= は使わない
    resp = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
    )
    raw = resp.read()  # bytes（デフォルトは MP3 データ）

    if fmt == "mp3":
        return raw, "audio/mp3"

    # WAV が選択された場合はローカル変換
    try:
        seg = AudioSegment.from_file(io.BytesIO(raw), format="mp3")
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        return buf.getvalue(), "audio/wav"
    except Exception:
        # 変換失敗時は MP3 を返す
        return raw, "audio/mp3"

# -----------------------------
# UI サイドバー
# -----------------------------
with st.sidebar:
    st.header("⚙️ 設定 / Cài đặt")
    mode = st.radio("モード / Chế độ", ["テキスト翻訳 / Dịch văn bản", "音声入力 / Ghi âm", "会話モード / Hội thoại"], index=0) or "テキスト翻訳 / Dịch văn bản"
    st.divider()
    st.subheader("翻訳設定 / Cấu hình dịch")
    col1, col2 = st.columns(2)
    with col1:
        src_choice = st.selectbox("入力言語 / Ngôn ngữ nguồn", ["auto", "vi", "ja"], index=0) or "auto"
    with col2:
        dst_choice = st.selectbox("出力言語 / Ngôn ngữ đích", ["ja", "vi"], index=0) or "ja"
    st.caption("Tip: 'auto'=自動判定 / tự động phát hiện")

    st.divider()
    st.subheader("音声設定 / Cấu hình giọng nói")
    tts_voice = st.selectbox("音声タイプ / Giọng", ["alloy", "verse", "aria", "sage"], index=0) or "alloy"
    audio_format = st.selectbox("音声形式 / Định dạng", ["mp3", "wav"], index=0) or "mp3"

# -----------------------------
# 各モード (UI 表示も日越併記)
# -----------------------------
if mode.startswith("テキスト"):
    st.subheader("📝 テキスト翻訳 / Dịch văn bản")
    example = "Xin chào, rất vui được gặp bạn." if dst_choice == "ja" else "今日はとても暑いですね。"
    text_in = st.text_area("テキスト入力 / Nhập văn bản", example, height=150)
    if st.button("翻訳 / Dịch", type="primary"):
        if not text_in.strip():
            st.warning("テキストを入力してください / Vui lòng nhập văn bản")
        else:
            with st.spinner("翻訳中... / Đang dịch..."):
                out = translate_text(text_in, src_choice, dst_choice)
            st.success("完了 / Hoàn tất")
            st.markdown("**翻訳結果 / Kết quả**")
            st.text_area("", out, height=150)
            audio_bytes, mime = speak(out, voice=tts_voice, fmt=audio_format)
            if audio_bytes:
                st.audio(audio_bytes, format=mime)

elif mode.startswith("音声入力"):
    st.subheader("🎤 音声入力翻訳 / Dịch giọng nói")
    st.caption("クリックして録音 / Nhấn để ghi âm")

    wav_bytes = audio_recorder(text="録音 / Ghi âm", recording_color="#e53935", neutral_color="#6c757d", icon_size="2x")
    if wav_bytes:
        st.info("録音完了 / Đã ghi âm. テキスト化中... / Đang nhận dạng...")
        transcript = transcribe_bytes(wav_bytes, src_choice if src_choice != "auto" else "auto")
        st.markdown("**文字起こし / Văn bản**")
        st.write(transcript)

        with st.spinner("翻訳中... / Đang dịch..."):
            out = translate_text(transcript, src_choice, dst_choice)
        st.markdown("**翻訳 / Bản dịch**")
        st.write(out)

        audio_bytes, mime = speak(out, voice=tts_voice, fmt=audio_format)
        if audio_bytes:
            st.audio(audio_bytes, format=mime)

elif mode.startswith("会話"):
    st.subheader("🗣️ 会話モード / Hội thoại")
    st.caption("交互に話してください / Nói lần lượt")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    wav_bytes = audio_recorder(text="話す / Nói", recording_color="#1e88e5", neutral_color="#6c757d", icon_size="2x")
    if wav_bytes:
        transcript = transcribe_bytes(wav_bytes, "auto")
        detected = detect_lang_simple(transcript)
        target = "ja" if detected == "vi" else "vi"
        translation = translate_text(transcript, detected, target)
        st.session_state.chat.append({
            "speaker": "A" if (len(st.session_state.chat) % 2 == 0) else "B",
            "transcript": transcript,
            "translation": translation,
            "src": detected,
            "dst": target,
        })
        audio_bytes, mime = speak(translation, voice=tts_voice, fmt=audio_format)
        if audio_bytes:
            st.audio(audio_bytes, format=mime)

    for i, msg in enumerate(reversed(st.session_state.chat)):
        role = msg["speaker"]
        st.markdown(f"### {len(st.session_state.chat)-i} 回目 / Lượt {len(st.session_state.chat)-i} · 話者 / Người nói {role}")
        st.markdown(f"**原文 ({msg['src']})**: {msg['transcript']}")
        st.markdown(f"**翻訳 ({msg['dst']})**: {msg['translation']}")
        st.divider()

# -----------------------------
# Footer
# -----------------------------
st.caption("❤️ Streamlit + OpenAI で構築 · Xây dựng bằng Streamlit và OpenAI · FFmpeg 推奨 / Nên cài FFmpeg")