import os
import os.path
import random
import shutil
import argparse
import subprocess

from ftplib import FTP_TLS, error_perm
from zipfile import ZipFile

files = []

def get_file_list(dir_output):
    for line in dir_output.splitlines():
        l = line.split()
        files.append((l[-1], l[4]))

def get_downloaded_list():
    downloaded = []
    for f in os.listdir('./downloads/'):
        if os.path.isfile('./downloads/'+f) and f.endswith('.zip'):
            downloaded.append((f, 0))
    return downloaded

def in_downloads(downloaded, filename):
    for f, s in downloaded:
        if f == filename: return True
    return False

def download(downloaded, all_files=False, filename=None):
    # Connect to the MAPS ftp server over FTPS
    ftps = FTP_TLS('ftps.tsi.telecom-paristech.fr')
    print 'Connected to MAPS FTP over TLS.'
    try:
        ftps.login(user='jdb7hw@virginia.edu', passwd='tFcvL2CA')
        ftps.cwd('maps')
    except error_perm:
        print "Incorrect username/password" ; quit

    ftps.retrlines('LIST *.zip', get_file_list)

    if filename is not None:
        if not in_downloads(files, filename): print 'File not found' ; return
        print 'Downloading', filename
        res = ftps.retrbinary('RETR '+filename, open('./downloads/'+filename, 'wb').write)
        ftps.close()
        return [(filename, 0)]
    
    if len(files) == len(downloaded):
        print "All MAPS files downloaded. Continuing."
        return
    
    if all_files:
        for f, s in files:
            if not in_downloads(downloaded, f):
                print "Downloading", f, "of size", s, "bytes"
                res = ftps.retrbinary('RETR '+f, open('./downloads/'+f, 'wb').write)
    elif filename is None:
        f, s = random.choice(files)
        while in_downloads(downloaded, f):
            f, s = random.choice(files)
        
        print "Downloading", f, "of size", s, "bytes"
        res = ftps.retrbinary('RETR '+f, open('./downloads/'+f, 'wb').write)

    ftps.close()

    if all_files: return files
    return [(f, s)]

def merge_directories(downloaded):
    for d, _ in downloaded:
        key = d.split('_')[1]
        print 'Merging ./data/' + key + '.'
        cwd = os.getcwd()
        p = subprocess.Popen(['cp -R ./data/'+key+'/* ./data'], shell=True)
        p.wait()
        shutil.rmtree('./data/' + key)
    
if __name__ == '__main__':
    # Set up our command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', dest='download_all', action='store_true',
                        help='Boolean flag for downloading all of MAPS')
    parser.add_argument('--new', dest='download_new', action='store_true',
                        help='Boolean flag for downloading new files')
    parser.add_argument('-f', dest='filename', help='Name a specific file to download')
    
    args = parser.parse_args()
    allf = args.download_all
    newf = args.download_new

    # Check for previous downloads
    downloaded = get_downloaded_list()
    
    if args.filename is not None:
        download([], filename=args.filename)
    else:
        if len(downloaded) > 0 and not newf: print 'Detected MAPS files download.'
        else: downloaded = download(downloaded, allf)

    # Unzip all our downloaded files
    for d, _ in downloaded:
        key = d.split('_')[1]
        if os.path.isdir('./data/'+key): print 'Detected', key, 'already unzipped.' ; continue
        with ZipFile('./downloads/'+d, 'r') as z:
            print 'Unzipping', './downloads/' + d, 'into ./data.'
            z.extractall('./data/')
    
    # Check if data/ directory exists
    if not os.path.isdir('./data'): print 'Creating data/ directory.' ; os.makedirs('./data')
    else: print 'Directory data/ already exists.'
            
    # Fix merge all the directories
    merge_directories(downloaded)

    to_remove = ['license.html', 'licence.html', 'license.txt', 'licence.txt', 'readme.txt']
    print 'Cleaning up...'
    for r in to_remove:
        try:
            os.remove('./data/'+r)
        except OSError:
            continue

    print "Done!"
