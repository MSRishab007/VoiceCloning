from speaker import BaseSpeaker, ToneConverter
from AudioExtract import get_se
import os
import torch
import soundfile

device="cpu"
output_directory = "output"
os.makedirs(output_directory, exist_ok=True)

base_speaker_model_path =r"C:\Users\Hp\OneDrive\Desktop\S.Rishab\Projects\VoiceCloning\VoiceCloningModel\Dependencies\checkpoints\base_speakers\EN"
tone_converter_model_path = r"C:\Users\Hp\OneDrive\Desktop\S.Rishab\Projects\VoiceCloning\VoiceCloningModel\Dependencies\checkpoints\converter"

base_speaker = BaseSpeaker(f"{base_speaker_model_path}\\config.json",device=device)
base_speaker.load_model(f"{base_speaker_model_path}\\checkpoint.pth")

tone_converter = ToneConverter(f"{tone_converter_model_path}\\config.json",device=device)
tone_converter.load_model(f"{tone_converter_model_path}\\checkpoint.pth")

source_se=torch.load(f"{base_speaker_model_path}/en_default_se.pth").to(device)
reference_speaker=r"C:\Users\Hp\Downloads\tate_original.mp3"
target_se,audio_name=get_se(reference_speaker,tone_converter,target_dir="Temporary",vad=True)

final_audio_path = os.path.join(output_directory, f"{audio_name}.wav")

text="The speech had been delivered in 1986 by Richard Hamming, an accomplished mathematician and computer engineer, as part of an internal series of talks given at Bell Labs. I had never heard of Hamming, the internal lecture series at Bell Labs, or this particular speech. And yet, as I read the transcript, I came across one useful insight after another."
base_speaker_output_path=f"{output_directory}/temporary.wav"
base_speaker.text_to_speech(text,base_speaker_output_path,speaker="default",language="English",sampling_rate=22050,speed=1.0)

watermark="MiniProject"
tone_converter.convert_tone(audio_source_path=base_speaker_output_path,source_se=source_se,target_se=target_se,output_path=final_audio_path,message=watermark)