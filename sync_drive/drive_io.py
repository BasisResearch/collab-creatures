# upload file to shared google drive folder
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def upload_to_google_drive(drive, to_upload, shared_drive_folderid, skip_if_exists=False):
    # split to_upload into filename and foldername
    filename = to_upload.split('/')[-1]
    local_foldername = '/'.join(to_upload.split('/')[:-1])
    # if it's a file, upload it
    if os.path.isfile(to_upload):
        upload_file_to_google_drive(drive, filename, local_foldername, shared_drive_folderid, skip_if_exists=skip_if_exists)
    else:
        # make google drive subfolder if it doesn't exist
        if len(drive.ListFile({'q': "title='" + filename + "' and mimeType='application/vnd.google-apps.folder' and trashed=false and '" + shared_drive_folderid + "' in parents"}).GetList()) == 0:
            folder = drive.CreateFile({'title': filename, 'parents': [{'id': shared_drive_folderid}], 'mimeType': 'application/vnd.google-apps.folder'})
            folder.Upload()
        else:
            folder = drive.ListFile({'q': "title='" + filename + "' and mimeType='application/vnd.google-apps.folder' and trashed=false and '" + shared_drive_folderid + "' in parents"}).GetList()[0]
        # upload contents of folder
        folderid = folder['id']
        for file in os.listdir(to_upload):
            # check if it's a folder
            if os.path.isdir(os.path.join(to_upload, file)):
                upload_to_google_drive(drive, os.path.join(to_upload, file), folderid, skip_if_exists=skip_if_exists)
            else:
                upload_file_to_google_drive(drive, file, to_upload, folderid, skip_if_exists=skip_if_exists)
    


def upload_file_to_google_drive(drive, filename, local_foldername, shared_drive_folderid, skip_if_exists=False):
    # if file already exists, delete it
    file_list = drive.ListFile({'q': "title='" + filename + "' and trashed=false"}).GetList()
    if len(file_list) > 0:
        if skip_if_exists:
            return
        else:
            file_list[0].Trash()
    file = drive.CreateFile({'title': filename, 'parents': [{'id': shared_drive_folderid}]})
    file.SetContentFile(os.path.join(local_foldername, filename))
    file.Upload()
    return


def ls_google_drive(drive, shared_drive_folderid): 
    file_list = drive.ListFile({'q': "'" + shared_drive_folderid + "' in parents and trashed=false"}).GetList()
    for file in file_list:
        print('%s, id: %s' % (file['title'], file['id']))

def get_google_drive_id(drive, filename, shared_drive_folderid):
    file_list = drive.ListFile({'q': "'" + shared_drive_folderid + "' in parents and trashed=false"}).GetList()
    for file in file_list:
        if file['title'] == filename:
            return file['id']
    return None

def get_root_id(drive, shared_drive_foldername):
    shared_drive_folder = drive.ListFile({'q': "title='" + shared_drive_foldername + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()[0]
    return shared_drive_folder['id']

def is_drive_folder(drive, foldername, shared_drive_foldername):
    shared_drive_folderid = get_root_id(drive, shared_drive_foldername)
    if len(drive.ListFile({'q': "title='" + foldername + "' and trashed=false and '" + shared_drive_folderid + "' in parents and mimeType='application/vnd.google-apps.folder'"}).GetList()) == 0:
        return False
    else:
        return True


def download_file_from_google_drive(drive, local_foldername, shared_drive_id, skip_if_exists=False):
    file = drive.CreateFile({'id': shared_drive_id})
    filename = file['title']  
    # if file already exists, skip
    if os.path.exists(os.path.join(local_foldername, filename)):
        if skip_if_exists:
            return
    file.GetContentFile(os.path.join(local_foldername, filename))


def download_from_google_drive(drive, local_foldername, shared_drive_id, skip_if_exists=False):
    file = drive.CreateFile({'id': shared_drive_id})
    filename = file['title']  
    # if it's a file, download it
    if file['mimeType'] != 'application/vnd.google-apps.folder':
        download_file_from_google_drive(drive, local_foldername, file['id'], skip_if_exists=skip_if_exists)
    else:
        # if it's a folder, download all files in it
        if not os.path.exists(os.path.join(local_foldername, filename)):
            os.makedirs(os.path.join(local_foldername, filename))
        file_list = drive.ListFile({'q': "'" + shared_drive_id + "' in parents and trashed=false"}).GetList()
        for file in file_list:
            download_from_google_drive(drive, os.path.join(local_foldername, filename), file['id'], skip_if_exists=skip_if_exists)

def pull_from_collab_drive(shared_drive_foldername, local_repo_foldername, data_foldername, skip_if_exists=False):
    # authenticate drive
    drive = get_drive()

    # download from google drive
    shared_drive_id = get_root_id(drive, shared_drive_foldername)
    data_folder_id = get_google_drive_id(drive, data_foldername, shared_drive_id)
    download_from_google_drive(drive, local_repo_foldername, data_folder_id, skip_if_exists=skip_if_exists)
    print(f"Downloaded {data_foldername} from {shared_drive_foldername} to {local_repo_foldername}")

def push_to_collab_drive(shared_drive_foldername, local_repo_foldername, data_foldername, skip_if_exists=False):
    # authenticate drive
    drive = get_drive()

    # upload to google drive
    shared_drive_id = get_root_id(drive, shared_drive_foldername)
    upload_to_google_drive(drive, os.path.join(local_repo_foldername, data_foldername), shared_drive_id, skip_if_exists=skip_if_exists)
    print(f"Uploaded {data_foldername} from {local_repo_foldername} to {shared_drive_foldername}")

def get_drive(): 
    # authenticate drive
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive