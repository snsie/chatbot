class BaseSpeaker:
    async def speak(self, sentence: str):  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self):  # optional cleanup
        pass


class CoquiTTSSpeaker(BaseSpeaker):
    """Coqui TTS speaker using neural voice synthesis.
    
    Uses TTS library for high-quality neural text-to-speech synthesis.
    Enhanced with better audio processing and prosody control.
    """

    def __init__(self, model_name: Optional[str] = VOICE_NAME):
        self.model_name = model_name or "tts_models/en/vctk/vits"  # Better multi-speaker model
        try:
            from TTS.api import TTS
            import torch
            import sounddevice as sd
            import tempfile
            import os
            import wave
        except ImportError as e:
            print("[CoquiTTSSpeaker] Missing packages. Install: pip install TTS torch torchaudio", file=sys.stderr)
            raise
        
        # Initialize TTS model with GPU acceleration
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CoquiTTS] Loading model {self.model_name} on {self.device}...")
        
        try:
            self.tts = TTS(model_name=self.model_name).to(self.device)
            print(f"[CoquiTTS] Successfully loaded on {self.device}")
        except Exception as e:
            print(f"[CoquiTTS] Failed to load {self.model_name} on {self.device}, trying fallback...")
            # Try CPU fallback if GPU fails
            if self.device == "cuda":
                self.device = "cpu"
                print(f"[CoquiTTS] Falling back to CPU...")
                try:
                    self.tts = TTS(model_name=self.model_name).to(self.device)
                except Exception as e2:
                    print(f"[CoquiTTS] CPU fallback failed, trying simpler model...")
                    self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"
                    self.tts = TTS(model_name=self.model_name).to(self.device)
            else:
                print(f"[CoquiTTS] Trying simpler model on CPU...")
                self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"
                self.tts = TTS(model_name=self.model_name).to(self.device)
        
        # Check if model supports speaker selection
        if hasattr(self.tts, 'speakers') and self.tts.speakers:
            print(f"[CoquiTTS] Available speakers: {self.tts.speakers[:3]}...")  # Show first 3
            self.speaker_name = self.tts.speakers[0]
        else:
            self.speaker_name = None
        
        print(f"[CoquiTTS] Model loaded successfully. Speaker: {self.speaker_name or 'default'}")

    def _enhance_text(self, sentence: str) -> str:
        """Add prosody and cleanup to make speech more natural."""
        # Clean up text
        sentence = sentence.strip()
        if not sentence:
            return sentence
            
        # Add slight pauses for better pacing
        sentence = sentence.replace(',', ', ')  # Pause after commas
        sentence = sentence.replace(';', '; ')  # Pause after semicolons
        
        # Emphasize questions and exclamations
        if sentence.endswith('?'):
            sentence = sentence[:-1] + '?'  # Ensure proper question intonation
        elif sentence.endswith('!'):
            sentence = sentence[:-1] + '!'  # Ensure proper exclamation intonation
            
        return sentence

    async def speak(self, sentence: str):
        import sounddevice as sd
        import numpy as np
        import torch
        
        # Enhance text for better prosody
        enhanced_sentence = self._enhance_text(sentence)
        if not enhanced_sentence.strip():
            return
        
        try:
            # Use GPU acceleration for synthesis
            with torch.no_grad():  # Disable gradients for inference
                if self.device == "cuda":
                    # GPU-optimized synthesis
                    torch.cuda.empty_cache()  # Clear cache before synthesis
                
                # Generate audio with proper parameters
                if self.speaker_name:
                    wav = self.tts.tts(text=enhanced_sentence, speaker=self.speaker_name)
                else:
                    wav = self.tts.tts(text=enhanced_sentence)
                
                # Move to CPU for audio processing if on GPU
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                elif not isinstance(wav, np.ndarray):
                    wav = np.array(wav, dtype=np.float32)
            
            # Normalize audio to prevent clipping
            if len(wav) > 0:
                max_val = np.max(np.abs(wav))
                if max_val > 0:
                    wav = wav / max_val * 0.85  # Leave some headroom
            
            # Get sample rate from TTS model
            if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'output_sample_rate'):
                sample_rate = getattr(self.tts.synthesizer.output_sample_rate, 'value', 22050)
            else:
                sample_rate = 22050
            
            # Play using sounddevice with optimized settings
            sd.play(wav, samplerate=sample_rate, blocking=True)
            
            # Clear GPU memory after synthesis
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[CoquiTTS] Synthesis error: {e}")
            # Clear GPU memory on error too
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            # Fallback: skip this sentence
            pass

    async def close(self):
        """Clean up GPU resources when done."""
        if hasattr(self, 'device') and self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                print("[CoquiTTS] GPU memory cleared")
            except:
                pass
