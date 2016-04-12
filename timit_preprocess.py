from scipy.io import wavfile
import subprocess
import glob
import os.path

frame_size=0.05
hop_size=0.05

def fix_wavs():
    for d in ['TRAIN', 'TEST']:
        timit_files = glob.glob('./data/TIMIT/{0}/*/*/*.WAV'.format(d))
        wav_files = []

        for f in timit_files:
            name, ext = os.path.split(f)
            name += '_tmp'
            wav_files.append(name+ext)

        sx_cmd = 'sox {0} -t wav {1}'
        mv_cmd = 'mv {0} {1}'
        
        for i, f in enumerate(timit_files):
            subprocess.call(sx_cmd.format(f, wav_files[i]), shell=True)
            os.remove(f)
            subprocess.call(mv_cmd.format(wav_files[i], f), shell=True)

    with open('./data/TIMIT/corrected.txt', 'w+') as f:
        f.write('True')

if __name__ == '__main__':
    if not os.path.isfile('./data/TIMIT/corrected.txt'):
        print "Fixing wav files..."
        fix_wavs()
    else: print "Wav files already corrected..."

    s = Stairway(False)\
        .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
        .step('transform', ['load_audio'], stft, frame_size, hop_size)

    files = r_load_pairs('./data/TIMIT/TRAIN', exts=['.wav', '.phn'], master='.phn')

    with h5py.File('timit.h5', 'w') as hf:
        X = hf.create_dataset('X', (), maxshape=(), dtype='float32')
        y = hf.create_dataset('y', (), maxshape=(), dtype='float32')

        cnt = 0
        for f in files:
            print 'Preprocessing:', f[0]
            
