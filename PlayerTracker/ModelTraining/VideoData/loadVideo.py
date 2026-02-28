import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
import os
from dotenv import load_dotenv

load_dotenv()

mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=os.getenv("SOCCERNET_DATASET_PATH"))
mySoccerNetDownloader.password = os.getenv("SOCCERNET_PASSWORD")

mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])
