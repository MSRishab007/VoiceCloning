from TextExtract import cleaned_text_english
import torch
import numpy as np
import re
import soundfile
from Dependencies.Models import commons
import os
import librosa
from Dependencies.Models.mel_processing import spectrogram_torch
from Dependencies.Models.models import SynthesizerTrn
from Dependencies.configurationset import base_speaker_hps, tone_converter_hps
from Dependencies.utilityfunctions import get_hyper_parameters_from_file, string_to_bits, bits_to_string    
from AudioExtract import get_se
import wavmark

speakers={
    "default": 1,
    "whispering": 2,
    "shouting": 3,
    "excited": 4,
    "cheerful": 5,
    "terrified": 6,
    "angry": 7,
    "sad": 8,
    "friendly": 9
  }
class ParentClass:
    def __init__(self,parameter_path,device="cpu"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        parameters=get_hyper_parameters_from_file(parameter_path)
        model = SynthesizerTrn(
            len(getattr(parameters, 'symbols', [])),
            parameters.data.filter_length // 2 + 1,
            n_speakers=parameters.data.n_speakers,
            **parameters.model,
        ).to(device)
        model.eval()
        self.model = model
        self.parameters = parameters
        self.device = device
    def load_model(self,model_path):
        checkpoint_dict = torch.load(model_path, map_location=torch.device(self.device), weights_only=True)
        a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print("Loaded checkpoint '{}'".format(model_path))
        print('missing/unexpected keys:', a, b)

class BaseSpeaker(ParentClass):
    def text_to_speech(self,text,output_path,speaker="default",language="English",speed=1.0,sampling_rate=22050):
        if language!="English":
            raise ValueError("Language not supported")
        list_of_sequences=cleaned_text_english(text)
        speaker_id=speakers[speaker]
        device=self.device
        audio_list=[]
        for t in list_of_sequences:
            with torch.no_grad():
                k=t.unsqueeze(0).to(device)
                lenk=torch.LongTensor([t.size(0)]).to(device)   
                speaker_id_tensor=torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(k, lenk, sid=speaker_id_tensor,noise_scale=0.667, noise_scale_w=0.6,
                                    length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
            audio_list.append(audio)
        audio_segements=[]
        for audio in audio_list:
            audio_segements+=audio.reshape(-1).tolist()
            audio_segements+=[0]*int((sampling_rate*0.05)/speed)
        audio_segements=np.array(audio_segements).astype(np.float32)
        soundfile.write(output_path, audio_segements, sampling_rate)


class ToneConverter(ParentClass):
    def extract_speaker_embeddings(self,reference_wav_list,se_save_path=None):
        if isinstance(reference_wav_list,str):
            reference_wav_list=[reference_wav_list]
        device=self.device
        parameters=self.parameters
        gs=[]
        for i in reference_wav_list:
            audio_reference,sr=librosa.load(i,sr=parameters.data.sampling_rate)
            audioy=torch.FloatTensor(audio_reference)
            audioy=audioy.to(device)
            audioy=audioy.unsqueeze(0)
            audioy=spectrogram_torch(audioy,parameters.data.filter_length,parameters.data.sampling_rate,parameters.data.hop_length,parameters.data.win_length,center=False).to(device)  
            with torch.no_grad():
                g=self.model.ref_enc(audioy.transpose(1,2)).unsqueeze(-1) 
                gs.append(g.detach())
        gs=torch.stack(gs).mean(0)
        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs.cpu(),se_save_path)
        return gs
    
    def convert_tone(self,audio_source_path,source_se,target_se,output_path=None,tau=0.3,message=None):
        device=self.device
        parameters=self.parameters
        audio,sr=librosa.load(audio_source_path,sr=parameters.data.sampling_rate)
        audio=torch.FloatTensor(audio)
        with torch.no_grad():
            audio=torch.FloatTensor(audio).to(device)
            audio=audio.unsqueeze(0)
            spec=spectrogram_torch(audio,parameters.data.filter_length,parameters.data.sampling_rate,parameters.data.hop_length,parameters.data.win_length,center=False).to(device)
            spec_length=torch.LongTensor([spec.size(-1)]).to(device)
            audio=self.model.voice_conversion(spec,spec_length,sid_src=source_se,sid_tgt=target_se,tau=tau)[0][0,0].data.cpu().float().numpy()
            audio = self.add_watermark(audio, message)
            if output_path is None:
                return audio
            else:
                soundfile.write(output_path, audio, parameters.data.sampling_rate)
    
    def add_watermark(self,audio,message):
        self.watermark_model = wavmark.load_model().to(self.device)
        device = self.device
        bits = string_to_bits(message).reshape(-1)
        n_repeat = len(bits) // 32

        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K: (coeff * n + 1) * K]
            if len(trunck) != K:
                print('Audio too short, fail to add watermark')
                break
            message_npy = bits[n * 32: (n + 1) * 32]
            
            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(device)[None]
                signal_wmd_tensor = self.watermark_model.encode(signal, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().squeeze()
            audio[(coeff * n) * K: (coeff * n + 1) * K] = signal_wmd_npy
        return audio
    
    def detect_watermark(self, audio, n_repeat):
        bits = []
        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K: (coeff * n + 1) * K]
            if len(trunck) != K:
                print('Audio too short, fail to detect watermark')
                return 'Fail'
            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(self.device).unsqueeze(0)
                message_decoded_npy = (self.watermark_model.decode(signal) >= 0.5).int().detach().cpu().numpy().squeeze()
            bits.append(message_decoded_npy)
        bits = np.stack(bits).reshape(-1, 8)
        message = bits_to_string(bits)
        return message
    
    

        