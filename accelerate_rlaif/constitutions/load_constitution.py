import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import archieml
import json

load_dotenv()

DOCUMENT_ID = "1S5W-4-NXlptG0ngRnmoFj5rF4qYM2QGSxLrsU7U5z8w"
TAB_ID = "t.35mpwvjdf4bn"
SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]

def load_credentials():
    
    client_secret_file = os.getenv("CLIENT_SECRET_FILE")
    creds = None
    
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secret_file, SCOPES
            )
            creds = flow.run_local_server(port=59366)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds

load_credentials()

def extract_text_from_tab(tab):
    text_chunks = []

    content = tab["documentTab"]["body"]["content"]

    for element in content:
        if "paragraph" not in element:
            continue

        for run in element["paragraph"].get("elements", []):
            text_run = run.get("textRun")
            if text_run:
                text_chunks.append(text_run.get("content", ""))

    return "".join(text_chunks)

def get_document_text(document_id, tab_id, creds):
    try:
        service = build("docs", "v1", credentials=creds)
        document = service.documents().get(documentId=document_id,
                                           includeTabsContent=True).execute()
        tabs = document.get("tabs", [])
                
        for tab in tabs:
            if tab['tabProperties']['tabId'] == tab_id:
                print(f"Found tab with ID: {tab_id}")
                print(tab)
                content = extract_text_from_tab(tab)

                return content
            
    except HttpError as error:
        print(f"An error occurred: {error}")

creds = load_credentials()

document_text = get_document_text(DOCUMENT_ID, TAB_ID, creds)

data = archieml.loads(document_text)
output_file = "constitution_from_doc.json"

with open(output_file, "w") as json_file:
    json.dump(data, json_file, indent=4)