#!/usr/bin/env python3
"""
WhisperX Audio Transcription and Speaker Diarization Script
Manages VRAM usage by sequentially loading and unloading models for under 16GB GPU usage.

Usage:
    python transcribe_and_diarize.py <path_to_video_or_audio> [--hf_token <token>] [--batch_size <size>]
    
Example:
    python transcribe_and_diarize.py video.mov --hf_token hf_xxxxxxx
"""

import os
import sys
import gc
import argparse
import logging
from pathlib import Path
import subprocess
from dotenv import load_dotenv

import torch
import whisperx
from pyannote.audio import Pipeline

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_audio(media_file, output_audio_path, sr=16000):
    """
    Extract audio from a media file using ffmpeg.
    
    Args:
        media_file (str): Path to the video/audio file
        output_audio_path (str): Path to save the extracted audio (WAV format)
        sr (int): Sample rate for audio extraction (default: 16000 Hz)
    
    Returns:
        str: Path to the extracted audio file
    """
    logger.info(f"Extracting audio from {media_file}...")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg is not installed. Please install it and try again.")
        sys.exit(1)
    
    # Extract audio using ffmpeg
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', media_file,
        '-acodec', 'pcm_s16le',
        '-ar', str(sr),
        '-ac', '1',
        '-y',
        output_audio_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        logger.info(f"Audio extracted successfully to {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio: {e}")
        sys.exit(1)


def clear_vram():
    """
    Clear GPU VRAM using garbage collection and PyTorch cache clearing.
    
    This function helps manage VRAM by:
    - Running Python garbage collection
    - Clearing PyTorch's CUDA cache
    """
    gc.collect()
    torch.cuda.empty_cache()
    logger.debug("VRAM cleared")


def get_device():
    """
    Determine the device to use for computation.
    
    Returns:
        str: 'cuda' if NVIDIA GPU is available, otherwise 'cpu'
    
    Logs GPU information including device name and available VRAM.
    """
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return "cuda"
    else:
        logger.warning("CUDA is not available. Using CPU. This will be significantly slower.")
        return "cpu"


def transcribe_audio(audio_path, device, batch_size=16, compute_type="float16"):
    """
    Transcribe audio using WhisperX large-v2 model.
    
    Args:
        audio_path (str): Path to the audio file
        device (str): Device to use ('cuda' or 'cpu')
        batch_size (int): Batch size for transcription (default: 16)
        compute_type (str): Compute type for model (default: 'float16')
    
    Returns:
        dict: Transcribed result with word-level timestamps
    
    Note:
        Model is automatically deleted after transcription to manage VRAM.
        VRAM is cleared using gc.collect() and torch.cuda.empty_cache().
    """
    logger.info("Loading WhisperX model (large-v2)...")
    
    try:
        # Load the model with specified compute type and batch size
        model = whisperx.load_model(
            "large-v2",
            device=device,
            compute_type=compute_type,
            language="en"
        )
        
        logger.info("Model loaded. Starting transcription...")
        
        # Transcribe with word-level timestamps
        result = model(audio_path, batch_size=batch_size, language="en")
        
        logger.info("Transcription completed")
        
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        sys.exit(1)
    finally:
        # Clean up the model and clear VRAM
        del model
        clear_vram()
    
    return result


def align_timestamps(audio_path, result, device):
    """
    Align word-level timestamps using WhisperX alignment model.
    
    Args:
        audio_path (str): Path to the audio file
        result (dict): Transcription result from whisperx
        device (str): Device to use ('cuda' or 'cpu')
    
    Returns:
        dict: Result with aligned timestamps
    
    Note:
        Alignment model is language-specific and automatically selected.
        Model is deleted after alignment to manage VRAM.
    """
    logger.info("Loading alignment model...")
    
    try:
        # Get model name and language from result
        language = result.get("language", "en")
        
        # Load alignment model
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        
        logger.info("Alignment model loaded. Aligning timestamps...")
        
        # Align the result
        result = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
        
        logger.info("Alignment completed")
        
    except Exception as e:
        logger.error(f"Error during alignment: {e}")
        logger.warning("Continuing without alignment...")
    finally:
        # Clean up the model and clear VRAM
        del model_a
        clear_vram()
    
    return result


def diarize_audio(audio_path, hf_token, device):
    """
    Diarize audio to identify speakers using pyannote.audio.
    
    Args:
        audio_path (str): Path to the audio file
        hf_token (str): Hugging Face API token (required)
        device (str): Device to use ('cuda' or 'cpu')
    
    Returns:
        dict: Diarization result with speaker labels, or None if token not provided
    
    Note:
        Requires Hugging Face token with access to pyannote models.
        Pipeline is deleted after diarization to manage VRAM.
    """
    if not hf_token:
        logger.warning("Hugging Face token not provided. Skipping diarization.")
        return None
    
    logger.info("Loading pyannote diarization pipeline...")
    
    try:
        # Load the diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=hf_token
        )
        
        # Move pipeline to device
        pipeline = pipeline.to(torch.device(device))
        
        logger.info("Pipeline loaded. Starting diarization...")
        
        # Perform diarization
        diarization = pipeline(audio_path)
        
        logger.info("Diarization completed")
        
    except Exception as e:
        logger.error(f"Error during diarization: {e}")
        logger.warning("Continuing without diarization...")
        return None
    finally:
        # Clean up and clear VRAM
        del pipeline
        clear_vram()
    
    return diarization


def assign_speakers_to_words(result, diarization):
    """
    Assign speaker labels to transcribed words based on diarization results.
    
    Args:
        result (dict): Aligned transcription result with word-level timestamps
        diarization: Diarization result from pyannote, or None
    
    Returns:
        dict: Result with speaker labels assigned to words
    
    Note:
        If diarization is None, all words are assigned to SPEAKER_1.
        Speaker assignment is based on word midpoint timestamp.
    """
    if diarization is None:
        logger.warning("No diarization data available. Assigning all text to SPEAKER_1")
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                word["speaker"] = "SPEAKER_1"
        return result
    
    logger.info("Assigning speakers to words...")
    
    # Create speaker mapping
    speaker_map = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"SPEAKER_{len(speaker_map) + 1}"
    
    # Assign speakers to words based on timing
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            if "start" in word and "end" in word:
                word_start = word["start"]
                word_end = word["end"]
                word_mid = (word_start + word_end) / 2
                
                # Find the speaker at this time
                assigned = False
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= word_mid <= turn.end:
                        word["speaker"] = speaker_map.get(speaker, "UNKNOWN")
                        assigned = True
                        break
                
                if not assigned:
                    word["speaker"] = "UNKNOWN"
            else:
                word["speaker"] = "UNKNOWN"
    
    logger.info("Speaker assignment completed")
    return result


def export_to_txt(result, output_file):
    """
    Export transcription with speaker labels to a text file.
    
    Args:
        result (dict): Final aligned and diarized result with speaker labels
        output_file (str): Path to save the output text file
    
    Output format:
        SPEAKER_ID: Transcribed text
        
    Note:
        Consecutive words from the same speaker are combined into single lines.
    """
    logger.info(f"Exporting results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        current_speaker = None
        current_text = []
        
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                speaker = word.get("speaker", "UNKNOWN")
                text = word.get("word", "").strip()
                
                if not text:
                    continue
                
                # If speaker changes, write the previous speaker's text
                if speaker != current_speaker and current_text:
                    f.write(f"{current_speaker}: {' '.join(current_text)}\n")
                    current_text = []
                    current_speaker = None
                
                # Add text to current speaker
                if speaker != current_speaker:
                    current_speaker = speaker
                    current_text = [text]
                else:
                    current_text.append(text)
        
        # Write the last speaker's text
        if current_text and current_speaker:
            f.write(f"{current_speaker}: {' '.join(current_text)}\n")
    
    logger.info(f"Results exported to {output_file}")


def main():
    """
    Main function to orchestrate the transcription and diarization workflow.
    
    Workflow:
        1. Extract audio from media file
        2. Transcribe audio with WhisperX
        3. Align word-level timestamps
        4. Perform speaker diarization
        5. Assign speakers to words
        6. Export results to text file
    
    Each model is loaded, used, deleted, and VRAM is cleared sequentially
    to maintain memory usage under 16GB.
    """
    parser = argparse.ArgumentParser(
        description="WhisperX Audio Transcription and Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_and_diarize.py video.mov --hf_token hf_xxxxxxx
  python transcribe_and_diarize.py audio.wav --batch_size 32
        """
    )
    
    parser.add_argument(
        "media_file",
        type=str,
        help="Path to the video or audio file to transcribe"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for transcription (default: 16). Increase for faster processing if VRAM allows."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output text file path (default: <media_file>.txt)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    media_file = Path(args.media_file)
    if not media_file.exists():
        logger.error(f"File not found: {media_file}")
        sys.exit(1)
    
    # Set output file path
    output_file = args.output or str(media_file.with_suffix('.txt'))
    
    # Get device
    device = get_device()
    
    # Create temporary audio file
    temp_audio = "temp_audio.wav"
    
    try:
        # Step 1: Extract audio
        audio_path = extract_audio(str(media_file), temp_audio)
        
        # Step 2: Transcribe with VRAM management
        logger.info("\n" + "="*50)
        logger.info("STEP 1: TRANSCRIPTION")
        logger.info("="*50)
        result = transcribe_audio(audio_path, device, batch_size=args.batch_size)
        
        # Step 3: Align timestamps
        logger.info("\n" + "="*50)
        logger.info("STEP 2: TIMESTAMP ALIGNMENT")
        logger.info("="*50)
        result = align_timestamps(audio_path, result, device)
        
        # Step 4: Diarize audio
        logger.info("\n" + "="*50)
        logger.info("STEP 3: SPEAKER DIARIZATION")
        logger.info("="*50)
        diarization = diarize_audio(audio_path, args.hf_token, device)
        
        # Step 5: Assign speakers to words
        logger.info("\n" + "="*50)
        logger.info("STEP 4: ASSIGNING SPEAKERS")
        logger.info("="*50)
        result = assign_speakers_to_words(result, diarization)
        
        # Step 6: Export to text file
        logger.info("\n" + "="*50)
        logger.info("STEP 5: EXPORTING RESULTS")
        logger.info("="*50)
        export_to_txt(result, output_file)
        
        logger.info("\n" + "="*50)
        logger.info("TRANSCRIPTION AND DIARIZATION COMPLETE!")
        logger.info("="*50)
        logger.info(f"Output saved to: {output_file}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary audio file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            logger.debug(f"Removed temporary audio file: {temp_audio}")


if __name__ == "__main__":
    main()
