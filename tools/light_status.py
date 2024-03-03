from typing import Any

from langchain_core.tools import BaseTool
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv('ha_url')
ACCESS_TOKEN = os.getenv('ha_api_key')


class LightSwitch(BaseTool):
    """Tool that can turn lights on or off by their entity_id."""

    name: str = "lightswitch"
    description: str = (
        "Turns lights on or off by their entity_id."
    )

    def _run(self, entity_id: Any, state='off') -> Any:
        self.call_homeassistant(entity_id, state)

    def call_homeassistant(self, entity_id, state):
        print(f"{entity_id} -> {state}")
        url = BASE_URL + ('/api/services/light/turn_on' if state == 'on' else '/api/services/light/turn_off')
        headers = {'cache-control': 'no-cache',
                   'content-type': 'application/json',
                   'authorization': ('Bearer %s' % ACCESS_TOKEN)
                   }
        data = {'entity_id': entity_id}
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200 or response.status_code == 201:
            print(response.text)
            return f"Successfully turned {state} {entity_id}."
        else:
            return f"Failed to turn {state} {entity_id}, status code: {response.status_code}"

class LightStatus(BaseTool):
    """Tool that enumerates all lights in the home and their current status."""

    name: str = "lightstatus"
    description: str = (
        "Enumerates all lights in the home and their current status."
    )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.call_homeassistant()

    def call_homeassistant(self):
        url = BASE_URL + '/api/states'
        headers = {'cache-control': 'no-cache',
                   'content-type': 'application/json',
                   'authorization': ('Bearer %s' % ACCESS_TOKEN)
                   }
        response = requests.request("GET", url, headers=headers)
        if response.status_code == 200:
            all_entities = response.json()
            lights = [entity for entity in all_entities if entity['entity_id'].startswith('light.')]

            return '\n'.join(self.iterate_lights(lights))
        else:
            return f"Failed to retrieve data, status code: {response.status_code}"

    def iterate_lights(self, lights):
        for light in lights:
            entity_id = light['entity_id']
            friendly_name = light['attributes'].get('friendly_name', 'No friendly name')
            state = light['state']
            yield f"entity_id: '{entity_id}', friendly_name: '{friendly_name}', state: '{state}'"
