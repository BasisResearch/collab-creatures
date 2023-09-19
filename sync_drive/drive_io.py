# upload file to shared google drive folder
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def upload_to_google_drive(drive, to_upload, shared_drive_folderid):
    # split to_upload into filename and foldername
    filename = to_upload.split('/')[-1]
    local_foldername = '/'.join(to_upload.split('/')[:-1])
    # if it's a file, upload it
    if os.path.isfile(to_upload):
        upload_file_to_google_drive(drive, filename, local_foldername, shared_drive_folderid)
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
                upload_to_google_drive(drive, os.path.join(to_upload, file), folderid)
            else:
                upload_file_to_google_drive(drive, file, to_upload, folderid)
    


def upload_file_to_google_drive(drive, filename, local_foldername, shared_drive_folderid):
    # if file already exists, delete it
    file_list = drive.ListFile({'q': "title='" + filename + "' and trashed=false"}).GetList()
    if len(file_list) > 0:
        file_list[0].Trash()
    file = drive.CreateFile({'title': filename, 'parents': [{'id': shared_drive_folderid}]})
    file.SetContentFile(os.path.join(local_foldername, filename))
    file.Upload()


# # upload file to subfolder of shared google drive folder
# def upload_to_google_drive_subfolder(drive, filename, local_foldername, shared_drive_foldername, subfoldername):
#     shared_drive_folderid = get_root_id(drive, shared_drive_foldername)
#     # create subfolder if it doesn't exist
#     if len(drive.ListFile({'q': "title='" + subfoldername + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()) == 0:
#         subfolder = drive.CreateFile({'title': subfoldername, 'parents': [{'id': shared_drive_folderid}], 'mimeType': 'application/vnd.google-apps.folder'})
#         subfolder.Upload()
#     else:
#         subfolder = drive.ListFile({'q': "title='" + subfoldername + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()[0]
#     subfolderid = subfolder['id']
#     # if file already exists in the subfolder, delete it
#     file_list = drive.ListFile({'q': "title='" + filename + "' and trashed=false and '" + subfolderid + "' in parents"}).GetList()
#     if len(file_list) > 0:
#         file_list[0].Trash()
    
#     file = drive.CreateFile({'title': filename, 'parents': [{'id': subfolderid}]})
#     file.SetContentFile(os.path.join(local_foldername, filename))
#     file.Upload()

# # upload folder to subfolder of shared google drive folder
# def upload_folder_to_google_drive_subfolder(drive, foldername, shared_drive_foldername, subfoldername):
#     shared_drive_folderid = get_root_id(drive, shared_drive_foldername)
#     # create subfolder if it doesn't exist
#     if len(drive.ListFile({'q': "title='" + subfoldername + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()) == 0:
#         subfolder = drive.CreateFile({'title': subfoldername, 'parents': [{'id': shared_drive_folderid}], 'mimeType': 'application/vnd.google-apps.folder'})
#         subfolder.Upload()
#     else:
#         subfolder = drive.ListFile({'q': "title='" + subfoldername + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()[0]
#     subfolderid = subfolder['id']
#     # create folder if it doesn't exist
#     destination_foldername = foldername.split('/')[-1]
#     if len(drive.ListFile({'q': "title='" + destination_foldername + "' and trashed=false and '" + subfolderid + "' in parents and mimeType='application/vnd.google-apps.folder'"}).GetList()) == 0:
#         folder = drive.CreateFile({'title': destination_foldername, 'parents': [{'id': subfolderid}], 'mimeType': 'application/vnd.google-apps.folder'})
#         folder.Upload()
#     else:
#         folder = drive.ListFile({'q': "title='" + destination_foldername + "' and trashed=false and '" + subfolderid + "' in parents and mimeType='application/vnd.google-apps.folder'"}).GetList()[0]
#     folderid = folder['id']
#     for file in os.listdir(foldername):
#         # check if it's a folder
#         if os.path.isdir(os.path.join(foldername, file)):
#             print('Folder ' + file + ' not uploaded because it is a folder')
#         else:
#             file_drive = drive.CreateFile({'title': file, 'parents': [{'id': folderid}]})
#             file_drive.SetContentFile(os.path.join(foldername,  file))
#             file_drive.Upload()

# # upload folder to shared google drive folder
# def upload_folder_to_google_drive(drive, foldername, shared_drive_foldername):
#     shared_drive_folderid = get_root_id(drive, shared_drive_foldername)
#     # create folder if it doesn't exist
#     destination_foldername = foldername.split('/')[-1]
#     if len(drive.ListFile({'q': "title='" + destination_foldername + "' and trashed=false and '" + shared_drive_folderid + "' in parents and mimeType='application/vnd.google-apps.folder'"}).GetList()) == 0:
#         folder = drive.CreateFile({'title': destination_foldername, 'parents': [{'id': shared_drive_folderid}], 'mimeType': 'application/vnd.google-apps.folder'})
#         folder.Upload()
#     else:
#         folder = drive.ListFile({'q': "title='" + destination_foldername + "' and trashed=false and '" + shared_drive_folderid + "' in parents and mimeType='application/vnd.google-apps.folder'"}).GetList()[0]
#     folderid = folder['id']
#     for file in os.listdir(foldername):
#         file_drive = drive.CreateFile({'title': file, 'parents': [{'id': folderid}]})
#         file_drive.SetContentFile(foldername + '/' + file)
#         file_drive.Upload()

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


def download_file_from_google_drive(drive, local_foldername, shared_drive_id):
    try:
        file = drive.CreateFile({'id': shared_drive_id})
        filename = file['title']  
        file.GetContentFile(os.path.join(local_foldername, filename))
        print(os.path.join(local_foldername, filename))
        print(file['mimeType'])
    except Exception as e:
        print(f"An error occurred: {e}")

def download_from_google_drive(drive, local_foldername, shared_drive_id):
    file = drive.CreateFile({'id': shared_drive_id})
    filename = file['title']  
    print(file['mimeType'])
    # if it's a file, download it
    if file['mimeType'] != 'application/vnd.google-apps.folder':
        download_file_from_google_drive(drive, local_foldername, file['id'])
    else:
        # if it's a folder, download all files in it
        if not os.path.exists(os.path.join(local_foldername, filename)):
            os.makedirs(os.path.join(local_foldername, filename))
        file_list = drive.ListFile({'q': "'" + shared_drive_id + "' in parents and trashed=false"}).GetList()
        for file in file_list:
            download_from_google_drive(drive, os.path.join(local_foldername, filename), file['id'])
