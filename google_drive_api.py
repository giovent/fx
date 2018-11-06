# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:06:36 2018

@author: AntonioVentilii
"""

import pandas as pd

from io import StringIO
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly'
SCOPES = 'https://www.googleapis.com/auth/drive'
SCOPES = 'https://www.googleapis.com/auth/drive.readonly'


FOLDER_ID = '1zuHv4J5o0a22qhLYXU_0Ra1UgDj51bHe'

CREDENTIALS_JSON = 'credentials.json'


class GoogleDriveService():
    
    def __init__(self, credentials_json, scopes=SCOPES):
        #TODO delete toke.json already there
        creds = self.associate_scopes(credentials_json, scopes)
        service = build('drive', 'v3', http=creds.authorize(Http()))
        self.service = service
    
    def create_token(self):
        store = file.Storage('token.json')
        creds = store.get()
        return store, creds
    
    def associate_scopes(self, credentials_json, scopes):
        store, creds = self.create_token()
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets('credentials.json', scopes)
            creds = tools.run_flow(flow, store)
        return creds

        


def list_files(service, query):
    response = service.files().list(q=query, pageSize=10, fields="nextPageToken, files(id, name)", spaces='drive').execute()
    page_token = -1
    while page_token:
        items = response.get('files', [])
        for item in items:
            file_id = item['id']
            file_name = item['name']
            ref_date = pd.to_datetime(file_name.split('.')[0].split('_')[2], yearfirst=True)
            print(u'{0} ({1})'.format(file_name, file_id))
            print(ref_date)
            #content = service.files().get_media(fileId=item['id']).execute()
            #with StringIO(str(content, 'utf-8')) as data:
                #df = pd.read_csv(data, index_col=0)
        page_token = response.get('nextPageToken', None)
        page_token = None



def select_files_from_google_drive(cross, exchange, start_date, end_date):
    date_range = list(pd.date_range(start_date, end_date, normalize=True))
    credentials_json = CREDENTIALS_JSON
    scopes = SCOPES
    drive_service = GoogleDriveService(credentials_json, scopes)
    service = drive_service.service
    folder_id = FOLDER_ID
    file_name = 'fx_{cross} {exchange} Curncy_'.format(cross=cross, exchange=exchange)
    q = "name contains '{file_name}' and mimeType='text/csv' and '{folder_id}' in parents".format(file_name=file_name, folder_id=folder_id)
    df = pd.DataFrame()
    page_token = -1
    while page_token:
        if page_token==-1:
            page_token = None
        response = service.files().list(
                q=q,
                spaces='drive',
                #pageSize=10,
                fields="nextPageToken, files(id, name)",
                pageToken=page_token).execute()
        items = response.get('files', [])
        for item in items:
            file_id = item['id']
            file_name = item['name']
            ref_date = pd.to_datetime(file_name.split('.')[0].split('_')[2], yearfirst=True)
            if ref_date in date_range:
                print(u'{0} ({1})'.format(file_name, file_id))
                content = service.files().get_media(fileId=item['id']).execute()
                with StringIO(str(content, 'utf-8')) as data:
                    df = df.append(pd.read_csv(data, index_col=0))
        page_token = response.get('nextPageToken', None)
    return df





#cross = 'XBTUSD'
#exchange = 'BGN'
#
#
#start_date = pd.to_datetime('2000-01-01')
#end_date = pd.to_datetime('today')
#
#
#start_date = pd.to_datetime('2018-08-01')
#
#
#
#
#df = select_files_from_google_drive(cross, exchange, start_date, end_date)


