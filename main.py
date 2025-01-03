import os
import pandas as pd
from dotenv import load_dotenv
from GroomingDetector import GroomingDetector
  
def main():
    # Load the environment variables
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    # Create the GroomingDetector object
    groom_detector = GroomingDetector(qdrant_url, qdrant_key, temperature=0.5)
    # Load the test dataset
    df = pd.read_csv("csv_files/abusive_text_test.csv", delimiter=";", names=["CONVERSATION_ID", "AUTHORS_IDS", "IS_ABUSIVE", "CONVERSATION_TEXT"])
    # Test the GroomingDetector object
    for i in range(10):
        conv = df.iloc[i]['CONVERSATION_TEXT']
        if len(conv.split("|")) > 20:
            print(f"Ground-truth {i}:{bool(df.iloc[i]['IS_ABUSIVE'])} grooming\n")
            print(groom_detector.invoke(conv))
            print("\n ------------------- \n")

if __name__ == "__main__":
    main()
