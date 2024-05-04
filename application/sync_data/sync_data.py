import requests
import json


class Sync_data:
    def __init__(self, base_url: str, serial: str, username: str, password: str, cam_id: str):
        """
        Initializes an instance of the Sync_data class.

        Args:
            base_url (str): The base URL of the API.
            serial (str): The serial number.
            username (str): The username for authentication.
            password (str): The password for authentication.
            cam_id (str): The ID of the camera.
        """
        self.base_url = base_url
        self.authorization = None
        self.serial = serial
        self.username = username
        self.password = password
        self.cam_id = cam_id

    def get_token(self):
        """
        Returns the current authorization token.

        Returns:
            str: The authorization token.
        """
        return self.authorization

    def get_authentic_token(self):
        """
        Sends a POST request to the authentication endpoint to obtain an authorization token.
        """
        endpoint = "auth/token"
        payload = json.dumps({
            "serial": self.serial,
            "username": self.username,
            "password": self.password
        })

        headers = {
            'Content-Type': 'application/json'
        }

        url = self.base_url + endpoint

        response = requests.request("POST", url, headers=headers, data=payload)
        self.authorization = response.text

    def get_rule_config(self):
        """
        Sends a GET request to retrieve rule configuration for the specified camera ID.

        Returns:
            str: The rule configuration.
        """
        endpoint = f"rule-config/{self.cam_id}"

        payload = {}
        headers = {
            "Authorization": self.authorization
        }

        url = self.base_url + endpoint
        response = requests.request("GET", url, headers=headers, data=payload)
        return response.text

    def post_violation(self):
        """
        Placeholder method for posting violations.
        """
        pass
