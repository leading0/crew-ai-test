from langchain_core.tools import BaseTool
import requests
import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_URL = None
if os.getenv('ha_url', None) and os.getenv('ha_weather'):
    WEATHER_URL = os.getenv('ha_url') + os.getenv('ha_weather')

WEATHER_ACCESS_TOKEN = os.getenv('ha_api_key', None)

class WeatherReport(BaseTool):
    """Tool that gets the weather report."""

    name: str = "weatherreport"
    description: str = (
        "Get information about the current weather in your current location, as well as a forecast."
    )

    def _run(
            self,
            *args,
            **kwargs
    ) -> str:
        """Use the tool."""
        if WEATHER_URL and WEATHER_ACCESS_TOKEN:
            return self.call_homeassistant()
        else:
            print('returning fallback data because either WEATHER_URL or WEATHER_ACCESS_TOKEN is empty')
            return self.get_fallback()

    def call_homeassistant(self):
        url = WEATHER_URL
        headers = {'cache-control': 'no-cache',
                   'content-type': 'application/json',
                   'authorization': ('Bearer %s' % WEATHER_ACCESS_TOKEN)
                   }
        response = requests.request("GET", url, headers=headers)
        content = response.text
        return content

    def get_fallback(self):
        with open("./example/weather.json") as f:
            return f.read()
