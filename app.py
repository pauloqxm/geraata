import streamlit as st
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import os
import librosa
from pydub import AudioSegment
import io
import time
import subprocess
import sys

# Configuração da página
st.set_page_config(
    page_title="Transcrição de Áudio - PT-BR",
    page_icon="🎙️",
    layout="wide"
)

# Título e descrição
st.title("🎙️ Transcrição de Áudio em Português Brasileiro")
st.markdown("""
Faça upload de um arquivo de áudio e obtenha a transcrição automática em português!
""")

# Função para verificar se FFmpeg está instalado
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Verifica FFmpeg no início
ffmpeg_available = check_ffmpeg()

if not ffmpeg_available:
    st.warning("""
    ⚠️ **FFmpeg não encontrado!**
    
    Para converter arquivos de áudio, é necessário instalar o FFmpeg:
    
    **Windows:**
    ```bash
    # Usando chocolatey
    choco install ffmpeg
    # Ou baixe do site oficial: https://ffmpeg.org/download.html
    ```
    
    **macOS:**
    ```bash
    brew install ffmpeg
    ```
    
    **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```
    
    **No Streamlit Cloud:** Adicione isso no arquivo `packages.txt`:
    ```
    ffmpeg
    ```
    """)

# Sidebar com configurações
st.sidebar.title("Configurações")

# Seleção do modelo
model_size = st.sidebar.selectbox(
    "Tamanho do Modelo:",
    ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
    index=2,
    help="Modelos maiores são mais precisos mas mais lentos"
)

# Configurações de transcrição
beam_size = st.sidebar.slider("Beam Size", 1, 10, 5)
best_of = st.sidebar.slider("Best Of", 1, 10, 5)
temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)
vad_filter = st.sidebar.checkbox("Filtro VAD", value=True, help="Detecção de atividade de voz")

# Função para carregar o modelo
@st.cache_resource
def load_model(model_size):
    try:
        # Use GPU se disponível, caso contrário use CPU
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
        except:
            device = "cpu"
            compute_type = "int8"
        
        st.sidebar.info(f"Usando: {device.upper()}")
        
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função alternativa para converter áudio usando librosa (quando FFmpeg não está disponível)
def convert_audio_librosa(input_file, output_path):
    """Converte áudio usando librosa quando FFmpeg não está disponível"""
    try:
        # Se for um arquivo upload do Streamlit
        if hasattr(input_file, 'read'):
            audio_data, sample_rate = librosa.load(io.BytesIO(input_file.read()), sr=16000, mono=True)
        else:
            audio_data, sample_rate = librosa.load(input_file, sr=16000, mono=True)
        
        # Salva o arquivo usando soundfile
        import soundfile as sf
        sf.write(output_path, audio_data, sample_rate)
        return output_path
    except Exception as e:
        st.error(f"Erro na conversão com librosa: {e}")
        return None

# Função para converter áudio para formato compatível
def convert_audio(input_file, output_format="wav", progress_bar=None, status_text=None):
    """Converte áudio para formato WAV com taxa de amostragem compatível"""
    try:
        if status_text:
            status_text.text("📥 Lendo arquivo de áudio...")
        
        # Se FFmpeg não estiver disponível, usa librosa
        if not ffmpeg_available:
            if status_text:
                status_text.text("🔄 Convertendo com librosa (FFmpeg não disponível)...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as temp_file:
                output_path = temp_file.name
            
            result_path = convert_audio_librosa(input_file, output_path)
            
            if progress_bar:
                progress_bar.progress(100)
            
            if status_text:
                status_text.text("✅ Conversão concluída com librosa!")
            
            return result_path
        
        # Se FFmpeg estiver disponível, usa pydub (mais robusto)
        if hasattr(input_file, 'read'):
            if progress_bar:
                progress_bar.progress(10)
            # Reinicia a posição do arquivo
            input_file.seek(0)
            audio = AudioSegment.from_file(io.BytesIO(input_file.read()))
        else:
            audio = AudioSegment.from_file(input_file)
        
        if progress_bar:
            progress_bar.progress(30)
        
        if status_text:
            status_text.text("🔄 Convertendo para mono e 16kHz...")
        
        # Converte para mono e 16kHz (recomendado para Whisper)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        if progress_bar:
            progress_bar.progress(60)
        
        if status_text:
            status_text.text("💾 Salvando arquivo convertido...")
        
        # Salva em arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as temp_file:
            audio.export(temp_file.name, format=output_format)
        
        if progress_bar:
            progress_bar.progress(100)
        
        if status_text:
            status_text.text("✅ Conversão concluída!")
            
        return temp_file.name
    except Exception as e:
        st.error(f"Erro na conversão do áudio: {e}")
        return None

# Função para transcrever áudio com progresso
def transcribe_audio(model, audio_path, progress_bar=None, status_text=None):
    """Transcreve o áudio usando faster-whisper"""
    try:
        if status_text:
            status_text.text("🎯 Iniciando transcrição...")
        
        segments, info = model.transcribe(
            audio_path,
            language="pt",
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            vad_filter=vad_filter
        )
        
        if status_text:
            status_text.text("📝 Processando segmentos de áudio...")
        
        # Coleta todos os segmentos
        transcriptions = []
        segments_list = list(segments)
        total_segments = len(segments_list)
        
        if progress_bar:
            progress_bar.progress(0)
        
        for i, segment in enumerate(segments_list):
            transcriptions.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })
            
            if progress_bar and total_segments > 0:
                progress = (i + 1) / total_segments
                progress_bar.progress(progress)
                
            if status_text and total_segments > 0:
                status_text.text(f"📝 Transcrevendo segmento {i+1}/{total_segments}...")
        
        if status_text:
            status_text.text("✅ Transcrição concluída!")
            
        return transcriptions, info
    except Exception as e:
        st.error(f"Erro na transcrição: {e}")
        return None, None

# Interface principal
uploaded_file = st.file_uploader(
    "Faça upload do arquivo de áudio",
    type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac', 'webm'],
    help="Formatos suportados: WAV, MP3, M4A, OGG, FLAC, AAC, WEBM"
)

# Carrega o modelo
with st.spinner("Carregando modelo de transcrição..."):
    model = load_model(model_size)

if model is not None and uploaded_file is not None:
    # Mostra informações do arquivo
    st.subheader("📄 Informações do Arquivo")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nome", uploaded_file.name)
    with col2:
        st.metric("Tipo", uploaded_file.type.split('/')[-1].upper())
    with col3:
        st.metric("Tamanho", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
    
    # Aviso sobre FFmpeg se necessário
    if not ffmpeg_available and not uploaded_file.name.lower().endswith('.wav'):
        st.warning("""
        ⚠️ **FFmpeg não encontrado - usando método alternativo**
        
        A conversão de áudio pode ser mais lenta e alguns formatos podem não funcionar perfeitamente.
        Para melhor experiência, instale o FFmpeg.
        """)
    
    # Botão para iniciar transcrição
    if st.button("🎯 Iniciar Transcrição", type="primary"):
        # Container para progresso
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            st.subheader("📊 Progresso do Processamento")
            overall_progress = st.progress(0)
            conversion_progress = st.progress(0)
            transcription_progress = st.progress(0)
            status_text = st.empty()
        
        try:
            # Atualiza progresso geral
            overall_progress.progress(10)
            status_text.text("📥 Preparando arquivo...")
            
            # Salva arquivo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_audio_path = tmp_file.name
            
            overall_progress.progress(20)
            
            # Sempre converte o áudio para garantir compatibilidade
            status_text.text("🔄 Convertendo formato de áudio...")
            converted_path = convert_audio(
                uploaded_file,  # Passa o arquivo original
                progress_bar=conversion_progress,
                status_text=status_text
            )
            
            if converted_path:
                audio_path = converted_path
                overall_progress.progress(50)
            else:
                st.error("Erro na conversão do áudio")
                os.unlink(temp_audio_path)
                st.stop()
            
            # Transcreve o áudio
            status_text.text("🎯 Iniciando transcrição...")
            start_time = time.time()
            
            segments, info = transcribe_audio(
                model, 
                audio_path,
                progress_bar=transcription_progress,
                status_text=status_text
            )
            
            end_time = time.time()
            overall_progress.progress(100)
            
            # Limpa arquivos temporários
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            if os.path.exists(converted_path):
                os.unlink(converted_path)
        
        except Exception as e:
            status_text.text("❌ Erro no processamento!")
            st.error(f"Erro durante o processamento: {e}")
            # Limpeza em caso de erro
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            if 'converted_path' in locals() and os.path.exists(converted_path):
                os.unlink(converted_path)
            st.stop()
        
        if segments and info:
            # Mostra estatísticas
            st.success(f"✅ Transcrição concluída em {end_time - start_time:.2f} segundos!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duração do Áudio", f"{info.duration:.2f}s")
            with col2:
                st.metric("Idioma Detectado", info.language.upper())
            with col3:
                st.metric("Confiança do Idioma", f"{info.language_probability*100:.1f}%")
            
            # Exibe a transcrição completa
            st.subheader("📝 Transcrição Completa")
            full_text = " ".join([segment['text'] for segment in segments])
            st.text_area("Texto transcrito:", full_text, height=200, key="full_text")
            
            # Exibe segmentos com timestamps
            st.subheader("⏱️ Transcrição com Timestamps")
            for i, segment in enumerate(segments, 1):
                with st.expander(f"Segmento {i} - {segment['start']:.2f}s a {segment['end']:.2f}s"):
                    st.write(segment['text'])
            
            # Opção para download
            st.subheader("💾 Download da Transcrição")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download como texto simples
                st.download_button(
                    label="📥 Baixar como TXT",
                    data=full_text,
                    file_name=f"transcricao_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Download com timestamps
                timestamp_text = ""
                for segment in segments:
                    timestamp_text += f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n"
                
                st.download_button(
                    label="⏱️ Baixar com Timestamps",
                    data=timestamp_text,
                    file_name=f"transcricao_timestamps_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )

# Seção de instruções
with st.expander("ℹ️ Instruções de Uso"):
    st.markdown("""
    ### Como usar:
    1. **Faça upload** de um arquivo de áudio nos formatos suportados
    2. **Ajuste as configurações** na barra lateral se necessário
    3. **Clique em 'Iniciar Transcrição'** para processar o áudio
    4. **Acompanhe o progresso** nas barras de progresso
    5. **Visualize e baixe** o resultado
    
    ### Requisitos:
    - **FFmpeg** (recomendado): Para melhor compatibilidade com formatos de áudio
    - **Sem FFmpeg**: Funciona com métodos alternativos, mas pode ser mais limitado
    
    ### Dicas:
    - Para melhor precisão, use áudios com boa qualidade de áudio
    - Modelos maiores ("medium", "large") são mais precisos mas mais lentos
    - O filtro VAD ajuda a remover silêncios desnecessários
    - Arquivos WAV geralmente têm melhor desempenho
    """)

# Rodapé
st.markdown("---")
st.markdown(
    "Desenvolvido com Streamlit e Faster-Whisper | "
    "Modelos de transcrição por OpenAI Whisper"
)

# Mensagem se nenhum arquivo foi carregado
if uploaded_file is None:
    st.info("👆 Faça upload de um arquivo de áudio para começar a transcrição!")
