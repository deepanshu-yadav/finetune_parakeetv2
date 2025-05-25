import nemo.collections.asr as nemo_asr
import os
import wget
# Load the model
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
file_url = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
file_name = "file.wav"
wget.download(file_url, file_name)

# Save the model to a local directory
model_dir = "parakeet-tdt-0.6b-v2"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "parakeet-tdt-0.6b-v2.nemo")
asr_model.save_to(model_path)

# Load the model from the local .nemo file
asr_model = nemo_asr.models.ASRModel.restore_from(model_path)
print(asr_model.cfg)
# Transcribe an audio file
transcriptions = asr_model.transcribe(["file.wav"])
print(transcriptions)
