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
    overwrite, remove_vocals, model_path, enabled_models, max_audio_length):

    if "mt3" in enabled_models:
        mt3 = InferenceHandler(model_path)
    if "pt" in enabled_models:
        pt = PianoTranscriptionInferenceHandler(model_path)
    voc_rem = None
    if (remove_vocals):
        voc_rem = VocalRemover("vocal_remover/models/baseline.pth")

    common_path = os.path.commonprefix(audio_path_list)
    if ((not os.path.exists(common_path)) or (not os.path.isdir(common_path))):
        common_path = os.path.dirname(common_path)

    for audio_path in audio_path_list:
        output_path = os.path.join(output_directory, os.path.relpath(audio_path, common_path))
        output_path = f"{os.path.splitext(output_path)[0]}"
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        midi_path_mt3 = f"{output_path}.mt3.mid"
        midi_path_pt = f"{output_path}.pt.mid"

        if (not overwrite 
            and ("mt3" not in enabled_models or os.path.exists(midi_path_mt3)) 
            and ("pt" not in enabled_models or os.path.exists(midi_path_pt))
        ):
            if ("mt3" in enabled_models):
                print(f'SKIPPING: "{midi_path_mt3}"')
            if ("pt" in enabled_models):
                print(f'SKIPPING: "{midi_path_pt}"')
        else:
            try:
                audio_length = librosa.get_duration(filename = audio_path)
                if max_audio_length is not None and audio_length > max_audio_length:
                    print(f'SKIPPING (too long): "{audio_path}"')
                else:
                    print(f'LOADING: "{audio_path}"')
                    audio, audio_sr = librosa.load(audio_path)
                    if (remove_vocals):
                        print(f'PREPROCESSING (removing vocals): "{audio_path}"')
                        audio, audio_sr = voc_rem.predict(audio, audio_sr)
                    if ("mt3" in enabled_models and not os.path.exists(midi_path_mt3)):
                        print(f'TRANSCRIBING (mt3): "{audio_path}"')
                        mt3.inference(audio, audio_sr, audio_path, outpath=midi_path_mt3)
                        print(f'SAVED: "{midi_path_mt3}"')
                    if ("pt" in enabled_models and not os.path.exists(midi_path_pt)):
                        print(f'TRANSCRIBING (pt): "{audio_path}"')
                        pt.inference(audio, audio_sr, audio_path, outpath=midi_path_pt)
                        print(f'SAVED: "{midi_path_pt}"')
            except Exception:
                print(traceback.format_exc())
                print("")
                print(f'FAILED: "{output_path}"')
                max_audio_length = audio_length

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
    return sorted(l, key=alphanum_key)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='*', type=str, default=["/input"],
        help="Input audio folders")

    parser.add_argument("--output-folder", type=str, default="/output",
        help="Output midi folder")

    parser.add_argument("--extensions", nargs='+', type=str, default=["mp3", "wav", "flac"],
        help="Input audio extensions")

    parser.add_argument("--overwrite", action="store_true",
        help="Overwrite output files")

    parser.add_argument("--disable-vocal-removal", action='store_true',
        help="Disable vocal removal preprocessing step")
        
    parser.add_argument("--model-path", type=str, default="./pretrained",
        help="Pretrained models for transcription")

    parser.add_argument("--enabled-models", type=str, default="mt3,pt",
        help="Models used for transcription [mt3,pt] (comma separed)")

    parser.add_argument("--max-audio-length", type=float, default=None,
        help="Maximal audio length")

    args = parser.parse_args()

    input_files = []
    for path in args.input:
        input_files.extend([p for e in args.extensions 
            for p in pathlib.Path(path).rglob("*." + e)])
    input_files = natural_sort(input_files)

    run_inference(input_files, args.output_folder, 
        overwrite = args.overwrite, 
        remove_vocals = not args.disable_vocal_removal,
        model_path = args.model_path, 
        enabled_models = [s.strip() for s in args.enabled_models.split(",")],
        max_audio_length = args.max_audio_length)
