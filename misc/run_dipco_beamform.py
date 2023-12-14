import os
from shutil import copy
import subprocess

TARGET = 'U01'
DATA_ROOT = '/proj/disney/Dipco/audio/eval'
BEAMFORMIT_PATH = '/home/nmehlman/disney/BeamformIt/output/'
TMP_DIR = '/home/nmehlman/disney/BeamformIt/temp'

def clear_dir(directory):

    file_list = os.listdir(directory)

    # Loop through the list and remove each file
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error while removing {file_path}: {e}")


os.chdir('/home/nmehlman/disney/BeamformIt')
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)
else:
    clear_dir(TMP_DIR) # Remove old array files

sessions = set([wav.split('_')[0] for wav in os.listdir(DATA_ROOT) if '.wav' in wav])

for sess in sessions: # For each meeting

    print(f'Running beamforming for session {sess}')
    files = [f for f in os.listdir(DATA_ROOT) if f.startswith(sess) and 'beamform' not in f]
    
    for file in files: # Copy target files to single dir
        if TARGET in file:
            copy(os.path.join(DATA_ROOT, file), TMP_DIR)

    beamformit_command = f"./do_beamforming.sh {TMP_DIR} dipco"

    # Run BeamformIt command and wait for it to finish
    completed_process = subprocess.run(beamformit_command, shell=True, text=True, capture_output=True)

    # Check the return code to see if the command was successful
    if completed_process.returncode != 0:
        print(completed_process.stdout)
        raise Exception
    
    # Copy output back to
    copy(os.path.join(BEAMFORMIT_PATH, 'dipco', 'dipco.wav'), os.path.join(DATA_ROOT, f'{sess}_{TARGET}.beamform.wav'))

    clear_dir(TMP_DIR) # Remove old array files

