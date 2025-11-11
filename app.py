import streamlit as st
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import os
import librosa
import io
import time
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

# Configuração da página com carregamento otimizado
st.set_page_config(
    page_title="Transcrição de Áudio - PT-BR",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS para melhorar performance
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .reportview-container {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Título e descrição
st.title("🎙️ Transcrição de Áudio em Português Brasileiro")
st.markdown("""
Faça upload de um arquivo de áudio e obtenha a transcrição automática em português!
""")

# Função para verificar se FFmpeg está instalado
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Verifica FFmpeg
ffmpeg_available = check_ffmpeg()

if not ffmpeg_available:
    with st.sidebar:
        st.warning("⚠️ FFmpeg não encontrado - usando método alternativo")

# Sidebar com configurações
with st.sidebar:
    st.title("Configurações")
    
    # Seleção do modelo
    model_size = st.selectbox(
        "Tamanho do Modelo:",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Modelos maiores são mais precisos mas mais lentos"
    )
    
    # Configurações de transcrição
    beam_size = st.slider("Beam Size", 1, 5, 2)
    best_of = st.slider("Best Of", 1, 5, 2)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)
    vad_filter = st.checkbox("Filtro VAD", value=True, help="Detecção de atividade de voz")

# Função para carregar o modelo com fallback
@st.cache_resource(show_spinner=False)
def load_model(model_size):
    try:
        # Limita o uso de memória
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            # Limita o uso de VRAM
            torch.cuda.set_per_process_memory_fraction(0.8)
        else:
            device = "cpu"
            compute_type = "int8"
        
        # Usa modelos menores se houver limitação de memória
        if model_size in ["large-v2", "large-v3"] and device == "cpu":
            model_size = "medium"
            st.sidebar.info("Usando modelo medium (large requer muita RAM)")
        
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root="./models"
        )
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo {model_size}: {str(e)}")
        # Tenta carregar modelo menor como fallback
        if model_size != "tiny":
            st.info("Tentando carregar modelo tiny como fallback...")
            try:
                model = WhisperModel(
                    "tiny",
                    device="cpu",
                    compute_type="int8"
                )
                return model
            except:
                pass
        return None

# Função otimizada para converter áudio
def convert_audio_optimized(_uploaded_file, progress_callback=None):
    """Converte áudio de forma otimizada"""
    try:
        if progress_callback:
            progress_callback(10, "📥 Lendo arquivo...")
        
        # Lê o arquivo diretamente com librosa (mais leve)
        audio_data, original_sr = librosa.load(
            io.BytesIO(_uploaded_file.read()),
            sr=None,
            mono=True
        )
        
        if progress_callback:
            progress_callback(40, "🔧 Convertendo amostragem...")
        
        # Converte para 16kHz se necessário
        if original_sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)
        
        if progress_callback:
            progress_callback(70, "💾 Salvando arquivo...")
        
        # Salva como WAV temporário
        import soundfile as sf
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, audio_data, 16000)
        
        if progress_callback:
            progress_callback(100, "✅ Conversão concluída!")
        
        return temp_path
        
    except Exception as e:
        st.error(f"Erro na conversão: {str(e)}")
        return None

# Função otimizada para transcrição
def transcribe_audio_optimized(_model, _audio_path, progress_callback=None):
    """Transcreve áudio de forma otimizada"""
    try:
        if progress_callback:
            progress_callback(0, "🎯 Iniciando transcrição...")
        
        segments, info = _model.transcribe(
            _audio_path,
            language="pt",
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            vad_filter=vad_filter,
            without_timestamps=False
        )
        
        if progress_callback:
            progress_callback(30, "📝 Processando segmentos...")
        
        # Processa segmentos em lotes para evitar memory leak
        transcriptions = []
        batch_size = 10
        current_batch = []
        
        for i, segment in enumerate(segments):
            current_batch.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
            
            # Atualiza progresso a cada lote
            if i % batch_size == 0 and progress_callback:
                progress = 30 + min(60, (i / 100) * 60)
                progress_callback(progress, f"📝 Processando... {i} segmentos")
        
        transcriptions = current_batch
        
        if progress_callback:
            progress_callback(100, "✅ Transcrição concluída!")
        
        return transcriptions, info
        
    except Exception as e:
        st.error(f"Erro na transcrição: {str(e)}")
        return None, None

# Interface principal
uploaded_file = st.file_uploader(
    "Faça upload do arquivo de áudio (máx. 50MB)",
    type=['wav', 'mp3', 'm4a'],
    help="Formatos suportados: WAV, MP3, M4A. Arquivos menores processam mais rápido."
)

# Limita tamanho do arquivo
if uploaded_file and uploaded_file.size > 50 * 1024 * 1024:
    st.error("⚠️ Arquivo muito grande! Por favor, use arquivos menores que 50MB.")
    st.stop()

# Carrega o modelo apenas quando necessário
if uploaded_file is not None:
    with st.spinner("🔄 Carregando modelo de transcrição..."):
        model = load_model(model_size)
    
    if model is None:
        st.error("❌ Não foi possível carregar o modelo de transcrição.")
        st.stop()

# Processamento principal
if uploaded_file is not None and model is not None:
    # Informações do arquivo
    st.subheader("📄 Informações do Arquivo")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nome", uploaded_file.name)
    with col2:
        st.metric("Tamanho", f"{uploaded_file.size / 1024 / 1024:.1f} MB")
    
    # Botão de transcrição
    if st.button("🎯 Iniciar Transcrição", type="primary", use_container_width=True):
        
        # Containers para progresso
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        def update_progress(progress, message):
            with progress_placeholder:
                st.progress(progress, text=message)
            status_placeholder.text(message)
        
        try:
            # Fase 1: Conversão
            update_progress(0, "🔄 Iniciando conversão de áudio...")
            audio_path = convert_audio_optimized(uploaded_file, update_progress)
            
            if not audio_path:
                st.error("❌ Falha na conversão do áudio")
                st.stop()
            
            # Fase 2: Transcrição
            start_time = time.time()
            segments, info = transcribe_audio_optimized(model, audio_path, update_progress)
            end_time = time.time()
            
            # Limpeza
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            if not segments:
                st.error("❌ Falha na transcrição do áudio")
                st.stop()
            
            # Resultados
            st.success(f"✅ Transcrição concluída em {end_time - start_time:.1f} segundos!")
            
            # Estatísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duração", f"{info.duration:.1f}s")
            with col2:
                st.metric("Idioma", info.language.upper())
            with col3:
                st.metric("Confiança", f"{info.language_probability*100:.0f}%")
            
            # Transcrição completa
            st.subheader("📝 Transcrição Completa")
            full_text = " ".join(segment['text'] for segment in segments)
            st.text_area("Texto transcrito:", full_text, height=150, key="transcription")
            
            # Segmentos com timestamps
            st.subheader("⏱️ Segmentos com Timestamps")
            for i, segment in enumerate(segments[:20]):  # Limita a 20 segmentos para performance
                with st.expander(f"Segmento {i+1} - {segment['start']:.1f}s a {segment['end']:.1f}s"):
                    st.write(segment['text'])
            
            if len(segments) > 20:
                st.info(f"📋 Mostrando os primeiros 20 de {len(segments)} segmentos")
            
            # Download
            st.subheader("💾 Download")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Baixar TXT",
                    full_text,
                    file_name=f"transcricao_{uploaded_file.name.split('.')[0]}.txt",
                    use_container_width=True
                )
            
            with col2:
                timestamp_text = "\n".join(
                    f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}" 
                    for s in segments
                )
                st.download_button(
                    "⏱️ Baixar com Timestamps",
                    timestamp_text,
                    file_name=f"transcricao_timestamps_{uploaded_file.name.split('.')[0]}.txt",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Erro durante o processamento: {str(e)}")
            st.info("💡 Dica: Tente usar um arquivo menor ou modelo tiny")

# Instruções
with st.expander("📖 Instruções de Uso"):
    st.markdown("""
    **Como usar:**
    1. Faça upload de um arquivo de áudio (até 50MB)
    2. Ajuste as configurações na sidebar se necessário
    3. Clique em 'Iniciar Transcrição'
    4. Aguarde o processamento
    5. Visualize e baixe o resultado

    **Dicas para melhor performance:**
    - Use arquivos WAV quando possível
    - Modelos menores (tiny, base) são mais rápidos
    - Arquivos curtos (< 10min) processam mais rápido
    - Evite múltiplas transcrições simultâneas

    **Formatos suportados:** WAV, MP3, M4A
    """)

# Rodapé
st.markdown("---")
st.markdown(
    "Desenvolvido com Streamlit + Faster-Whisper • "
    "[Problemas? Reduza o tamanho do arquivo ou use modelo tiny]"
)

if uploaded_file is None:
    st.info("👆 Faça upload de um arquivo de áudio para começar!")
