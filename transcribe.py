import os
import argparse
import pathlib
import warnings
import re
import librosa
import traceback

from inference import InferenceHandler
from vocal_remover import VocalRemover
from piano_transcription_inference.piano_transcription_inference import PianoTranscription 
from piano_transcription_inference.piano_transcription_inference import sample_rate as pt_sr

warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', message='will be removed in v5 of Transformers')

class PianoTranscriptionInferenceHandler:
    def __init__(self, model_path):
        self.transcriptor = PianoTranscription(checkpoint_path=
            os.path.join(model_path, "note_F1=0.9677_pedal_F1=0.9186.pth"))

    def inference(self, input_audio, audio_sr, audio_path, outpath=None):
        audio = librosa.to_mono(librosa.resample(input_audio, orig_sr=audio_sr, target_sr=pt_sr))
        if outpath is None:
            filename = audio_path.split('/')[-1].split('.')[0]
            outpath = f'./out/{filename}.mid'
        self.transcriptor.transcribe(audio, outpath)

def run_inference(audio_path_list, output_directory, 
    overwrite, remove_vocals, model_path):

    pt = PianoTranscriptionInferenceHandler(model_path)
    mt3 = InferenceHandler(model_path)
    voc_rem = None
    if (remove_vocals):
        voc_rem = VocalRemover("vocal_remover/models/baseline.pth")

    common_path = os.path.commonprefix(audio_path_list)
    if ((not os.path.exists(common_path)) or (not os.path.isdir(common_path))):
        common_path = os.path.dirname(common_path)

    for audio_path in audio_path_list:
        midi_path = os.path.join(output_directory, os.path.relpath(audio_path, common_path))
        midi_path = f"{os.path.splitext(midi_path)[0]}.mid"
        if not os.path.exists(os.path.dirname(midi_path)):
            os.makedirs(os.path.dirname(midi_path))

        midi_path_mt3 = "{}.mt3.{}".format(*os.path.splitext(midi_path))
        midi_path_pt = "{}.pt.{}".format(*os.path.splitext(midi_path))

        if (not overwrite and os.path.exists(midi_path_mt3) and os.path.exists(midi_path_pt)):
            print(f'SKIPPING: "{midi_path}"')    
        else:
            try:
                print(f'LOADING: "{audio_path}"')
                audio, audio_sr = librosa.load(audio_path)
                if (remove_vocals):
                    print(f'PREPROCESSING (removing vocals): "{audio_path}"')
                    audio, audio_sr = voc_rem.predict(audio, audio_sr)
                if (not os.path.exists(midi_path_mt3)):
                    print(f'TRANSCRIBING (mt3): "{audio_path}"')
                    mt3.inference(audio, audio_sr, audio_path, outpath=midi_path_mt3)
                    print(f'SAVED: "{midi_path_mt3}"')
                if (not os.path.exists(midi_path_pt)):
                    print(f'TRANSCRIBING (pt): "{audio_path}"')
                    pt.inference(audio, audio_sr, audio_path, outpath=midi_path_pt)
                    print(f'SAVED: "{midi_path_pt}"')
            except Exception:
                print(traceback.format_exc())
                print("")
                print(f'FAILED: "{midi_path}"')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
    return sorted(l, key=alphanum_key)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='*', type=str, default=["/input"],
        help='input audio folders')
    parser.add_argument("--output-folder", type=str, default="/output",
        help='output midi folder')
    parser.add_argument("--extensions", nargs='+', type=str,
        default=["mp3", "wav", "flac"],
        help='input audio extensions')
    parser.add_argument("--disable-vocal-removal", action='store_true',
        help='disable vocal removal preprocessing step')
    parser.add_argument("--overwrite", action="store_true",
        help='overwrite output files')
    parser.add_argument("--model-path", type=str, default="./pretrained")

    args = parser.parse_args()

    input_files = []
    for path in args.input:
        input_files.extend([p for e in args.extensions 
            for p in pathlib.Path(path).rglob("*." + e)])
    input_files = natural_sort(input_files)

    run_inference(input_files, args.output_folder, 
        overwrite = args.overwrite, 
        remove_vocals = not args.disable_vocal_removal,
        model_path = args.model_path),
