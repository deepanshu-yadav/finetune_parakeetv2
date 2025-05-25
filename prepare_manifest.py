import json
import librosa
import os
import wget
import tarfile
import random
from pathlib import Path
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define paths
base_dir = os.path.join(os.getcwd(), "data", "LJSpeech-1.1")
dataset_dir = os.path.join(base_dir, "LJSpeech-1.1")
metadata_file = os.path.join(dataset_dir, "metadata.csv")  # Path to LJSpeech metadata
input_audio_dir = os.path.join(dataset_dir, "wavs")  # Original 22.05 kHz WAVs
output_audio_dir = os.path.join(dataset_dir, "wavs_16k")  # Resampled 16 kHz WAVs
manifest_file = os.path.join(dataset_dir, "manifest.json")  # NeMo manifest file
os.makedirs(base_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(input_audio_dir, exist_ok=True)

# Step 1: Download the LJSpeech dataset if it doesn't exist
dataset_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
dataset_tar = os.path.join(base_dir, "LJSpeech-1.1.tar.bz2")

if not os.path.exists(metadata_file):
    print(f"Downloading LJSpeech dataset from {dataset_url}...")
    wget.download(dataset_url, dataset_tar)
    print("\nExtracting dataset...")
    with tarfile.open(dataset_tar, "r:bz2") as tar:
        tar.extractall(base_dir)
    print(f"Dataset extracted to {dataset_dir}")
else:
    print(f"Dataset already exists at {dataset_dir}")
   
os.makedirs(output_audio_dir, exist_ok=True)  # Ensure output directory exists
train_manifest = "train_manifest.json"
val_manifest = "val_manifest.json"
test_manifest = "test_manifest.json"
target_sr = 16000  # Target sample rate for Parakeet v2

# Set random seed for reproducibility
random.seed(42)

# Ensure the audio directories exist
if not os.path.exists(input_audio_dir):
    raise FileNotFoundError(f"Input audio directory {input_audio_dir} does not exist.")
if not os.path.exists(output_audio_dir):
    os.makedirs(output_audio_dir)

# Step 1: Resample audio to 16kHz if needed
def resample_audio():
    wav_files = [f for f in os.listdir(input_audio_dir) if f.endswith(".wav")]
    for wav_file in tqdm(wav_files, desc="Resampling audio"):
        input_path = os.path.join(input_audio_dir, wav_file)
        output_path = os.path.join(output_audio_dir, wav_file)
        
        # Skip if already resampled
        if os.path.exists(output_path):
            continue
        
        try:
            audio, sr = librosa.load(input_path, sr=None)
            if sr != target_sr:
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sf.write(output_path, audio_resampled, target_sr, subtype='PCM_16')
            else:
                # Copy file if already at target sample rate
                with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
        except Exception as e:
            print(f"Error resampling {wav_file}: {e}")

# Step 2: Read and validate metadata, create entries
def process_metadata():
    entries = []
    error_count = 0
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                print(f"Skipping empty line {line_number} in {metadata_file}")
                continue
            
            parts = line.split("|")
            if len(parts) < 2:
                print(f"Skipping malformed line {line_number}: {line}")
                continue
            
            audio_id, normalized_text, _ = parts
            audio_path = os.path.join(output_audio_dir, f"{audio_id}.wav")
            
            # Check if audio file exists
            if not os.path.exists(audio_path):
                print(f"Line {line_number}: Audio file {audio_path} not found, skipping.")
                continue
            
            # Calculate duration
            try:
                duration = librosa.get_duration(path=audio_path)
            except Exception as e:
                print(f"Line {line_number}: Error calculating duration for {audio_path}: {e}")
                continue
            
            # Clean text to remove non-ASCII characters
            cleaned_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
            
            # Normalize audio path to use forward slashes
            normalized_audio_path = str(Path(audio_path).as_posix())
            
            entries.append({
                "audio_filepath": normalized_audio_path,
                "text": cleaned_text,
                "duration": duration
            })
    
    if error_count > 0:
        print(f"Found {error_count} errors in {metadata_file}. Please review the logs.")
    
    return entries

# Step 3: Write manifest file
def write_manifest(entries, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    print(f"Created {filename} with {len(entries)} entries")

# Main execution
if __name__ == "__main__":
    # Resample audio
    resample_audio()
    
    # Process metadata and create entries
    entries = process_metadata()
    
    # Shuffle and split the dataset
    random.shuffle(entries)
    train_entries, temp_entries = train_test_split(entries, test_size=0.2, random_state=42)
    val_entries, test_entries = train_test_split(temp_entries, test_size=0.5, random_state=42)
    
    # Write manifest files
    write_manifest(train_entries, train_manifest)
    write_manifest(val_entries, val_manifest)
    write_manifest(test_entries, test_manifest)
    
    print(f"Total entries processed: {len(entries)}")
    print(f"Training set: {len(train_entries)} entries")
    print(f"Validation set: {len(val_entries)} entries")
    print(f"Test set: {len(test_entries)} entries")