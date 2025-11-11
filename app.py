# Transcrição de Áudio em PT-BR com Streamlit + faster-whisper
# -------------------------------------------------------------
# Requisitos para instalar no ambiente:
#   pip install -r requirements.txt
#   O requirements.txt já inclui pydub>=0.25.1, pyaudioop (para Python 3.13) e imageio-ffmpeg.
#
# Executar local:
#   streamlit run app.py
#
# Dicas de deploy:
#   • Em hosts sem ffmpeg no sistema, usamos imageio-ffmpeg embutido.
#   • Se o provedor for muito restritivo, prefira subir WAV/MP3.

import os
import io
import math
import time
import tempfile
from typing import List, Tuple

import streamlit as st
import pandas as pd

# IMPORTANTE: não importar pydub no topo em Python 3.13 para evitar erro de audioop.
# Faremos import preguiçoso dentro da função de conversão.
# from pydub import AudioSegment  # <- NÃO usar aqui
# import imageio_ffmpeg           # <- NÃO usar aqui

from faster_whisper import WhisperModel

# Tenta expor o ffmpeg empacotado (imageio-ffmpeg) no PATH, para que o faster-whisper consiga usar
try:
    import imageio_ffmpeg, os
    _ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = _ff
    os.environ["PATH"] = os.pathsep.join([os.path.dirname(_ff), os.environ.get("PATH", "")])
except Exception:
    pass

# -----------------------------------
# UI CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Transcrição de Áudio",
    page_icon="🎙️",
    layout="wide"
)

PRIMARY = "#2A4D9B"
PRIMARY_2 = "#1a326a"
TEXT = "#1f2937"
MUTED = "#6b7280"

st.markdown(f"""
<style>
:root {{
  --primary:{PRIMARY};
  --primary-2:{PRIMARY_2};
  --text:{TEXT};
  --muted:{MUTED};
}}

.reportview-container .main .block-container{{padding-top:1rem;padding-bottom:2rem;}}

h1, h2, h3, h4, h5 {{ color: var(--primary-2); }}

.sidebar .sidebar-content {{ background: #fff; }}

div.stDownloadButton > button {{
    background: var(--primary);
    color:#fff;
    border-radius: 10px;
}}

.stProgress > div > div > div > div {{ background-color: var(--primary); }}

/* Cartões simples */
.card {{
  background: linear-gradient(135deg, #ffffff 0%, #f5f7ff 100%);
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 1rem 1.25rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Helpers
# -----------------------------------

def format_ts(seconds: float) -> str:
    """HH:MM:SS,mmm para SRT"""
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(rows: List[dict]) -> str:
    out = []
    for i, r in enumerate(rows, start=1):
        out.append(str(i))
        out.append(f"{format_ts(r['start'])} --> {format_ts(r['end'])}")
        out.append(r['text'].strip())
        out.append("")
    return "\n".join(out)


def to_vtt(rows: List[dict]) -> str:
    out = ["WEBVTT", ""]
    for r in rows:
        s = format_ts(r['start']).replace(',', '.')
        e = format_ts(r['end']).replace(',', '.')
        out.append(f"{s} --> {e}")
        out.append(r['text'].strip())
        out.append("")
    return "\n".join(out)


def convert_to_wav_if_needed(input_bytes: bytes, filename: str) -> Tuple[str, str]:
    """Salva o upload num arquivo temporário. Se não for wav/mp3/m4a/flac/ogg, tenta converter para wav.
       Usa imageio-ffmpeg embutido. Retorna caminho do arquivo salvo e extensão."""
    suffix = os.path.splitext(filename)[1].lower()
    tmpdir = tempfile.mkdtemp()
    raw_path = os.path.join(tmpdir, filename)
    with open(raw_path, 'wb') as f:
        f.write(input_bytes)

    # Formatos suportados diretamente pelo faster-whisper (via ffmpeg) se houver no sistema
    if suffix in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"]:
        return raw_path, suffix

    # Import preguiçoso (evita erro de audioop em Python 3.13 se pyaudioop não estiver pronto)
    try:
        from pydub import AudioSegment
        import imageio_ffmpeg
        # Força o caminho do ffmpeg pela lib empacotada
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

        audio = AudioSegment.from_file(raw_path)
        wav_path = os.path.join(tmpdir, os.path.splitext(filename)[0] + ".wav")
        audio.export(wav_path, format="wav")
        return wav_path, ".wav"
    except Exception as e:
        # Se falhar a conversão, devolve o arquivo original.
        # O processo de transcrição ainda pode tentar abrir via ffmpeg do host.
        return raw_path, suffix


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, device: str, compute_type: str):
    return WhisperModel(model_name, device=device, compute_type=compute_type)


# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.header("Configurações")

# Bloqueia modelos muito grandes em ambientes com pouca memória
low_mem_default = True

model_choices = ["tiny", "base", "small", "medium", "large-v3"]
if low_mem_default:
    model_choices = ["tiny", "base", "small", "medium"]

model_name = st.sidebar.selectbox(
    "Modelo",
    model_choices,
    index=2 if "small" in model_choices else 1
)

device = st.sidebar.selectbox("Dispositivo", ["cpu", "cuda"], index=0)
compute_type = st.sidebar.selectbox("Precisão", ["int8", "int8_float16", "float16", "float32"], index=0)

st.sidebar.subheader("Parâmetros")
language_opt = st.sidebar.selectbox("Idioma", ["Detectar automaticamente", "pt", "en", "es", "fr"], index=1)
vad_filter = st.sidebar.checkbox("Ativar VAD (remoção de silêncio)", value=True)
beam_size = st.sidebar.slider("Beam size", 1, 10, 3)

st.sidebar.markdown("""
**Dicas**

• Em cloud com pouca RAM, prefira **small** ou **base** + `int8`.  
• O app injeta o ffmpeg empacotado automaticamente.  
• Se der erro ao iniciar a transcrição, tente reduzir o `beam size`.
""")

# -----------------------------------
# Main
# -----------------------------------
st.title("🎙️ Transcrição de Áudio Online")
st.markdown("Suba um arquivo de áudio e gere a transcrição em minutos.")

uploaded = st.file_uploader(
    "Selecione um arquivo de áudio ou vídeo",
    type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma", "mp4", "mkv", "mov"],
    accept_multiple_files=False
)

col_a, col_b, col_c = st.columns([1,1,1])

with col_a:
    start_btn = st.button("Transcrever")
with col_b:
    clear_btn = st.button("Limpar")
with col_c:
    demo_btn = st.button("Carregar demo")

if clear_btn:
    st.session_state.pop('results', None)
    st.rerun()

if demo_btn and 'results' not in st.session_state:
    st.info("Demo pronta. Use seu próprio áudio para melhores resultados.")

if start_btn:
    if not uploaded:
        st.warning("Envie um arquivo primeiro.")
    else:
        try:
            with st.status("Preparando o modelo", expanded=False) as status:
                model = load_model(model_name, device, compute_type)
                status.update(label="Convertendo áudio se necessário")
                file_bytes = uploaded.read()
                audio_path, ext = convert_to_wav_if_needed(file_bytes, uploaded.name)

            st.success("Tudo pronto. Iniciando transcrição.")

            progress = st.progress(0)
            t0 = time.time()

            lang = None if language_opt == "Detectar automaticamente" else language_opt

            segments_iter, info = model.transcribe(
                audio_path,
                language=lang,
                vad_filter=vad_filter,
                beam_size=beam_size,
                condition_on_previous_text=True
            )

            rows = []
            est = max(5.0, (getattr(info, 'duration', 60) or 60) * 0.3)

            # Barra de progresso baseada no tempo real processado
progress = progress  # já criado acima
prog_text = st.empty()

total_dur = float(getattr(info, 'duration', 0) or 0)

for seg in segments_iter:
    rows.append({
        'start': seg.start,
        'end': seg.end,
        'text': seg.text
    })

    if total_dur > 0:
        p = min(1.0, max(0.0, seg.end / total_dur))
        prog_text.text(f"{p*100:.1f}% • {seg.end:.1f}s de {total_dur:.1f}s")
    else:
        # Fallback se a duração não for conhecida
        p = min(1.0, (time.time() - t0) / est)
        prog_text.text(f"{p*100:.1f}% • processando…")

    progress.progress(p)

            df = pd.DataFrame(rows)
            st.session_state['results'] = {
                'rows': rows,
                'df': df,
                'info': {
                    'duration': getattr(info, 'duration', None),
                    'language': getattr(info, 'language', None),
                    'language_probability': getattr(info, 'language_probability', None),
                    'model': model_name,
                    'device': device,
                    'compute_type': compute_type
                }
            }
            st.toast("Transcrição concluída.")
        except RuntimeError as e:
            st.error("Memória insuficiente para carregar o modelo. Tente 'base' ou 'small' e 'int8'.")
            st.exception(e)
        except FileNotFoundError as e:
            st.error("Falha ao localizar o ffmpeg. Recarregue a página e tente novamente.")
            st.exception(e)
        except Exception as e:
            st.error("Houve um erro durante a execução.")
            st.exception(e)

# -----------------------------------
# Resultados
# -----------------------------------
if 'results' in st.session_state:
    res = st.session_state['results']
    rows = res['rows']
    df = res['df']

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dur = res['info'].get('duration')
        st.metric("Duração", f"{dur:.1f}s" if dur is not None else "-")
    with c2:
        st.metric("Segmentos", f"{len(rows)}")
    with c3:
        lang = res['info'].get('language') or 'auto'
        st.metric("Idioma", lang)
    with c4:
        st.metric("Modelo", res['info'].get('model', ''))

    st.markdown("### Texto contínuo")
    full_text = " ".join([r['text'].strip() for r in rows]).strip()
    st.text_area("Transcrição", full_text, height=200)

    st.markdown("### Segmentos com timestamps")
    st.dataframe(df, use_container_width=True)

    # Arquivos para download
    srt_txt = to_srt(rows)
    vtt_txt = to_vtt(rows)
    json_txt = df.to_json(orient='records', force_ascii=False)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            label="Baixar TXT",
            data=full_text,
            file_name="transcricao.txt",
            mime="text/plain"
        )
    with col2:
        st.download_button(
            label="Baixar SRT",
            data=srt_txt,
            file_name="legenda.srt",
            mime="text/plain"
        )
    with col3:
        st.download_button(
            label="Baixar VTT",
            data=vtt_txt,
            file_name="legenda.vtt",
            mime="text/vtt"
        )
    with col4:
        st.download_button(
            label="Baixar JSON",
            data=json_txt,
            file_name="segmentos.json",
            mime="application/json"
        )

    st.markdown("### Observações")
    st.markdown(
        "Acurácia depende da qualidade do áudio. Para máxima qualidade use o modelo large-v3. "
        "Se o provedor de hospedagem não tiver ffmpeg, usamos o binário do imageio-ffmpeg."
    )
